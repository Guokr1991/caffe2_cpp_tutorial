#include "caffe2/util/misc.h"

#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/core/operator_gradient.h>
#include "caffe2/utils/proto_utils.h"
#include "caffe2/zoo/keeper.h"

#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(layer, "",
                     "Name of the layer on which to split the model.");
CAFFE2_DEFINE_string(folder, "", "Folder with subfolders with images");

CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_int(train_runs, 100, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of training runs.");
CAFFE2_DEFINE_int(batch_size, 64, "Training batch size.");
CAFFE2_DEFINE_double(learning_rate, 1e-4, "Learning rate.");
CAFFE2_DEFINE_bool(reshape_output, false,
                   "Reshape output (necessary for squeeznet)");

#include "caffe2/util/cmd.h"

namespace caffe2 {

void run() {
  if (!cmd_init("Partial Retrain Example")) {
    return;
  }

  if (!FLAGS_model.size()) {
    std::cerr << "specify a model name using --model <name>" << std::endl;
    for (auto const &pair : keeper_model_lookup) {
      std::cerr << "  " << pair.first << std::endl;
    }
    return;
  }

  if (!FLAGS_folder.size()) {
    std::cerr << "specify a image folder using --folder <name>" << std::endl;
    return;
  }

  if (!FLAGS_layer.size()) {
    std::cerr << "specify a layer layer using --layer <name>" << std::endl;
    return;
  }

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "layer: " << FLAGS_layer << std::endl;
  std::cout << "image_dir: " << FLAGS_folder << std::endl;
  std::cout << "db_type: " << FLAGS_db_type << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test_runs: " << FLAGS_test_runs << std::endl;
  std::cout << "batch_size: " << FLAGS_batch_size << std::endl;
  std::cout << "learning_rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "reshape_output: " << FLAGS_reshape_output << std::endl;

  std::string layer_safe = FLAGS_layer;
  std::replace(layer_safe.begin(), layer_safe.end(), '/', '_');
  std::replace(layer_safe.begin(), layer_safe.end(), '.', '_');

  std::string model_safe = FLAGS_model;
  std::replace(model_safe.begin(), model_safe.end(), '/', '_');
  auto path_prefix =
      FLAGS_folder + '/' + '_' + model_safe + '_' + layer_safe + '_';
  std::string db_paths[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    db_paths[i] = path_prefix + name_for_run[i] + ".db";
  }

  std::cout << std::endl;

  auto load_time = -clock();
  std::vector<std::string> class_labels;
  std::vector<std::pair<std::string, int>> image_files;
  load_labels(FLAGS_folder, path_prefix, class_labels, image_files);

  std::cout << "load model.." << std::endl;
  NetDef full_init_model, full_predict_model;
  ModelUtil full(full_init_model, full_predict_model);
  NetDef init_model[kRunNum], predict_model[kRunNum];
  ModelUtil models[kRunNum] = {
      {init_model[kRunTrain], predict_model[kRunTrain]},
      {init_model[kRunTest], predict_model[kRunTest]},
      {init_model[kRunValidate], predict_model[kRunValidate]},
  };
  for (int i = 0; i < kRunNum; i++) {
    models[i].init.SetName(name_for_run[i] + "_init_model");
    models[i].predict.SetName(name_for_run[i] + "_predict_model");
  }
  Keeper(FLAGS_model).AddModel(full, true);

  full.predict.CheckLayerAvailable(FLAGS_layer);

  NetDef first_init_model, first_predict_model, second_init_model,
      second_predict_model;
  ModelUtil first(first_init_model, first_predict_model);
  ModelUtil second(second_init_model, second_predict_model);
  full.Split(FLAGS_layer, first, second, FLAGS_device != "cudnn");

  if (FLAGS_device != "cpu") {
    first.init.SetDeviceCUDA();
    first.predict.SetDeviceCUDA();
  }

  pre_process(image_files, db_paths, first.init.net, first.predict.net,
              FLAGS_db_type, FLAGS_batch_size, FLAGS_size_to_fit);
  load_time += clock();

  for (int i = 0; i < kRunNum; i++) {
    models[i].AddDatabaseOps(name_for_run[i], FLAGS_layer, db_paths[i],
                             FLAGS_db_type, FLAGS_batch_size);
  }
  second.CopyTrain(FLAGS_layer, class_labels.size(), models[kRunTrain]);
  copy_test_model(second.predict.net, models[kRunValidate].predict.net);
  copy_test_model(second.predict.net, models[kRunTest].predict.net);

  auto output = models[kRunTrain].predict.Output(0);
  if (FLAGS_reshape_output) {
    auto output_reshaped = output + "_reshaped";
    for (int i = 0; i < kRunNum; i++) {
      models[i].predict.AddReshapeOp(output, output_reshaped, {0, -1});
    }
    output = output_reshaped;
  }

  models[kRunTrain].AddTrainOps(output, FLAGS_learning_rate, FLAGS_optimizer);
  ModelUtil(second.predict.net, models[kRunValidate].predict.net)
      .AddTestOps(output);
  ModelUtil(second.predict.net, models[kRunTest].predict.net)
      .AddTestOps(output);

  if (FLAGS_device != "cpu") {
    for (int i = 0; i < kRunNum; i++) {
      models[i].init.SetDeviceCUDA();
      models[i].predict.SetDeviceCUDA();
    }
  }

  if (FLAGS_dump_model) {
    std::cout << models[kRunTrain].init.Short();
    std::cout << models[kRunTrain].predict.Short();
  }

  std::cout << std::endl;

  Workspace workspace("tmp");
  unique_ptr<caffe2::NetBase> predict_net[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    auto init_net = CreateNet(models[i].init.net, &workspace);
    init_net->Run();
    predict_net[i] = CreateNet(models[i].predict.net, &workspace);
  }

  clock_t train_time = 0;
  clock_t validate_time = 0;
  clock_t test_time = 0;

  auto last_time = clock();
  auto last_i = 0;

  std::cout << "training.." << std::endl;
  for (auto i = 1; i <= FLAGS_train_runs; i++) {
    train_time -= clock();
    predict_net[kRunTrain]->Run();
    train_time += clock();

    auto steps_time = (float)(clock() - last_time) / CLOCKS_PER_SEC;
    if (steps_time > 5 || i == FLAGS_train_runs) {
      auto iter = BlobUtil(*workspace.GetBlob("iter")).Get().data<int64_t>()[0];
      auto lr = BlobUtil(*workspace.GetBlob("lr")).Get().data<float>()[0];
      auto train_accuracy =
          BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      auto train_loss =
          BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
      validate_time -= clock();
      predict_net[kRunValidate]->Run();
      validate_time += clock();
      auto validate_accuracy =
          BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      std::cout << "step: " << iter << "  rate: " << lr
                << "  loss: " << train_loss << "  accuracy: " << train_accuracy
                << " | " << validate_accuracy
                << "  step_time: " << std::setprecision(3)
                << steps_time / (i - last_i) << "s" << std::endl;
      last_i = i;
      last_time = clock();
    }
  }

  std::cout << std::endl;

  std::cout << "testing.." << std::endl;
  for (auto i = 1; i <= FLAGS_test_runs; i++) {
    test_time -= clock();
    predict_net[kRunTest]->Run();
    test_time += clock();

    if (i % 10 == 0) {
      auto accuracy =
          BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
      auto loss = BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
      std::cout << "step: " << i << " loss: " << loss
                << " accuracy: " << accuracy << std::endl;
    }
  }

  NetDef deploy_init_model;  // the final initialization model
  ModelUtil deploy(deploy_init_model, full.predict.net);
  deploy.init.SetName("retrain_" + full.init.net.name());
  for (const auto &op : deploy.init.net.op()) {
    auto &output = op.output(0);
    auto blob = workspace.GetBlob(output);
    if (blob) {
      auto tensor = BlobUtil(*blob).Get();
      auto init_op = deploy.init.net.add_op();
      init_op->set_type("GivenTensorFill");
      auto arg1 = init_op->add_arg();
      arg1->set_name("shape");
      for (auto dim : tensor.dims()) {
        arg1->add_ints(dim);
      }
      auto arg2 = init_op->add_arg();
      arg2->set_name("values");
      const auto &data = tensor.data<float>();
      for (auto i = 0; i < tensor.size(); ++i) {
        arg2->add_floats(data[i]);
      }
      init_op->add_output(output);
    } else {
      deploy.init.net.add_op()->CopyFrom(op);
    }
  }

  WriteProtoToBinaryFile(deploy.init.net, path_prefix + "init_net.pb");
  WriteProtoToBinaryFile(deploy.predict.net, path_prefix + "predict_net.pb");
  auto init_size = std::ifstream(path_prefix + "init_net.pb",
                                 std::ifstream::ate | std::ifstream::binary)
                       .tellg();
  auto predict_size = std::ifstream(path_prefix + "predict_net.pb",
                                    std::ifstream::ate | std::ifstream::binary)
                          .tellg();
  auto model_size = init_size + predict_size;

  std::cout << std::endl;

  std::cout << std::setprecision(3)
            << "load: " << ((float)load_time / CLOCKS_PER_SEC)
            << "s  train: " << ((float)train_time / CLOCKS_PER_SEC)
            << "s  validate: " << ((float)validate_time / CLOCKS_PER_SEC)
            << "s  test: " << ((float)test_time / CLOCKS_PER_SEC)
            << "s  model: " << ((float)model_size / 1000000) << "MB"
            << std::endl;
}

}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
