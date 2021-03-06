#include "caffe2/util/misc.h"

#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/core/operator_gradient.h>
#include "caffe2/util/plot.h"
#include "caffe2/util/window.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/zoo/keeper.h"

#include "res/imagenet_classes.h"

CAFFE2_DEFINE_string(model, "", "Name of one of the pre-trained models.");
CAFFE2_DEFINE_string(folder, "", "Folder with subfolders with images");

CAFFE2_DEFINE_string(db_type, "leveldb", "The database type.");
CAFFE2_DEFINE_int(size_to_fit, 224, "The image file.");
CAFFE2_DEFINE_int(train_runs, 1000, "The of training runs.");
CAFFE2_DEFINE_int(test_runs, 50, "The of training runs.");
CAFFE2_DEFINE_int(batch_size, 64, "Training batch size.");
CAFFE2_DEFINE_double(learning_rate, 1e-4, "Learning rate.");

CAFFE2_DEFINE_bool(zero_one, false, "Show zero-one for batch.");
CAFFE2_DEFINE_bool(display, false,
                   "Show worst correct and incorrect classification.");
CAFFE2_DEFINE_bool(reshape_output, false,
                   "Reshape output (necessary for squeeznet)");

#include "caffe2/util/cmd.h"

namespace caffe2 {

void run() {
  if (!cmd_init("Full Train Example")) {
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

  std::cout << "model: " << FLAGS_model << std::endl;
  std::cout << "image_dir: " << FLAGS_folder << std::endl;
  std::cout << "db_type: " << FLAGS_db_type << std::endl;
  std::cout << "size_to_fit: " << FLAGS_size_to_fit << std::endl;
  std::cout << "train_runs: " << FLAGS_train_runs << std::endl;
  std::cout << "test_runs: " << FLAGS_test_runs << std::endl;
  std::cout << "batch_size: " << FLAGS_batch_size << std::endl;
  std::cout << "learning_rate: " << FLAGS_learning_rate << std::endl;
  std::cout << "zero_one: " << (FLAGS_zero_one ? "true" : "false") << std::endl;
  std::cout << "display: " << (FLAGS_display ? "true" : "false") << std::endl;
  std::cout << "reshape_output: " << FLAGS_reshape_output << std::endl;

  auto path_prefix = FLAGS_folder + '/' + '_';
  std::string db_paths[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    db_paths[i] = path_prefix + name_for_run[i] + ".db";
  }

  if (FLAGS_display) {
    superWindow("Full Train Example");
    moveWindow("undercertain", 0, 0);
    resizeWindow("undercertain", 300, 300);
    setWindowTitle("undercertain", "uncertain but correct");
    moveWindow("overcertain", 0, 300);
    resizeWindow("overcertain", 300, 300);
    setWindowTitle("overcertain", "certain but incorrect");
    moveWindow("accuracy", 300, 0);
    resizeWindow("accuracy", 500, 300);
    moveWindow("loss", 300, 300);
    resizeWindow("loss", 500, 300);
  }

  std::cout << std::endl;

  auto load_time = -clock();
  std::vector<std::string> class_labels;
  std::vector<std::pair<std::string, int>> image_files;
  load_labels(FLAGS_folder, path_prefix, class_labels, image_files);

  std::cout << "load model.." << std::endl;
  NetDef full_init_model, full_predict_model;
  Keeper(FLAGS_model).AddModel(full_init_model, full_predict_model, false);

  if (FLAGS_device == "cudnn") {
    NetUtil(full_init_model).SetEngineOps("CUDNN");
    NetUtil(full_predict_model).SetEngineOps("CUDNN");
  }

  if (FLAGS_dump_model) {
    std::cout << NetUtil(full_init_model).Short();
    std::cout << NetUtil(full_predict_model).Short();
  }

  NetDef init_model[kRunNum];
  NetDef predict_model[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    init_model[i].set_name(name_for_run[i] + "_init_model");
    predict_model[i].set_name(name_for_run[i] + "_predict_model");
  }

  pre_process(image_files, db_paths, FLAGS_db_type, FLAGS_size_to_fit);
  load_time += clock();

  for (int i = 0; i < kRunNum; i++) {
    ModelUtil(init_model[i], predict_model[i])
        .AddDatabaseOps(name_for_run[i], full_predict_model.external_input(0),
                        db_paths[i], FLAGS_db_type, FLAGS_batch_size);
  }
  copy_train_model(full_init_model, full_predict_model,
                   full_predict_model.external_input(0), class_labels.size(),
                   init_model[kRunTrain], predict_model[kRunTrain]);
  copy_test_model(full_predict_model, predict_model[kRunValidate]);
  copy_test_model(full_predict_model, predict_model[kRunTest]);

  auto output = predict_model[kRunTrain].external_output(0);
  if (FLAGS_reshape_output) {
    auto output_reshaped = output + "_reshaped";
    for (int i = 0; i < kRunNum; i++) {
      NetUtil(predict_model[i]).AddReshapeOp(output, output_reshaped, {0, -1});
    }
    output = output_reshaped;
  }

  ModelUtil(init_model[kRunTrain], predict_model[kRunTrain])
      .AddTrainOps(output, FLAGS_learning_rate, FLAGS_optimizer);
  ModelUtil(full_predict_model, predict_model[kRunValidate]).AddTestOps(output);
  ModelUtil(full_predict_model, predict_model[kRunTest]).AddTestOps(output);

  if (FLAGS_zero_one) {
    NetUtil(predict_model[kRunValidate])
        .AddZeroOneOp(output, "label");
  }
  if (FLAGS_display) {
    NetUtil(predict_model[kRunValidate])
        .AddShowWorstOp(output, "label",
                        full_predict_model.external_input(0));
  }

  if (FLAGS_display) {
    NetUtil(predict_model[kRunTrain])
        .AddTimePlotOp("accuracy", "iter", "accuracy", "train", 10);
    NetUtil(predict_model[kRunValidate])
        .AddTimePlotOp("accuracy", "iter", "accuracy", "test");
    NetUtil(predict_model[kRunTrain])
        .AddTimePlotOp("loss", "iter", "loss", "train", 10);
    NetUtil(predict_model[kRunValidate])
        .AddTimePlotOp("loss", "iter", "loss", "test");
  }

  if (FLAGS_device != "cpu") {
    for (int i = 0; i < kRunNum; i++) {
      NetUtil(init_model[i]).SetDeviceCUDA();
      NetUtil(predict_model[i]).SetDeviceCUDA();
    }
  }

  std::cout << std::endl;

  Workspace workspace("tmp");
  unique_ptr<caffe2::NetBase> predict_net[kRunNum];
  for (int i = 0; i < kRunNum; i++) {
    auto init_net = CreateNet(init_model[i], &workspace);
    init_net->Run();
    predict_net[i] = CreateNet(predict_model[i], &workspace);
  }

  clock_t train_time = 0;
  clock_t validate_time = 0;
  clock_t test_time = 0;

  auto last_time = clock();
  auto last_i = 0;
  auto sum_accuracy = 0.f, sum_loss = 0.f;

  std::cout << "training.." << std::endl;
  for (auto i = 1; i <= FLAGS_train_runs; i++) {
    train_time -= clock();
    predict_net[kRunTrain]->Run();
    train_time += clock();

    sum_accuracy +=
        BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
    sum_loss += BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];

    auto steps_time = (float)(clock() - last_time) / CLOCKS_PER_SEC;
    if (steps_time > 5 || i == FLAGS_train_runs) {
      auto iter = BlobUtil(*workspace.GetBlob("iter")).Get().data<int64_t>()[0];
      auto lr = BlobUtil(*workspace.GetBlob("lr")).Get().data<float>()[0];
      auto train_loss = sum_loss / (i - last_i),
           train_accuracy = sum_accuracy / (i - last_i);
      sum_loss = 0;
      sum_accuracy = 0;
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
  auto test_step = 10;
  for (auto i = 1; i <= FLAGS_test_runs; i++) {
    test_time -= clock();
    predict_net[kRunTest]->Run();
    test_time += clock();

    sum_accuracy +=
        BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
    sum_loss += BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];

    if (i % test_step == 0) {
      auto loss = sum_loss / test_step, accuracy = sum_accuracy / test_step;
      sum_loss = 0;
      sum_accuracy = 0;
      std::cout << "step: " << i << " loss: " << loss
                << " accuracy: " << accuracy << std::endl;
    }
  }

  NetDef deploy_init_model;  // the final initialization model
  deploy_init_model.set_name("train_" + full_init_model.name());
  for (const auto &op : full_init_model.op()) {
    auto &output = op.output(0);
    auto blob = workspace.GetBlob(output);
    if (blob) {
      auto tensor = BlobUtil(*blob).Get();
      auto init_op = deploy_init_model.add_op();
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
      deploy_init_model.add_op()->CopyFrom(op);
    }
  }

  auto init_path = path_prefix + FLAGS_model + "_init_net.pb";
  auto predict_path = path_prefix + FLAGS_model + "_predict_net.pb";
  WriteProtoToBinaryFile(deploy_init_model, init_path);
  WriteProtoToBinaryFile(full_predict_model, predict_path);
  auto init_size =
      std::ifstream(init_path, std::ifstream::ate | std::ifstream::binary)
          .tellg();
  auto predict_size =
      std::ifstream(predict_path, std::ifstream::ate | std::ifstream::binary)
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
