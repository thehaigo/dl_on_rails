class ResNet18Train < ActiveType::Object
  EX = Chainer::Training::Extensions
  attribute :class_label, :integer, default: 10
  attribute :learning_rate, :float, default: 0.05
  attribute :batch_size, :integer, default: 64
  attribute :epoch, :interger, default: 300
  attribute :out, :string, default: "./lib/weights/weights/resnet18.weights"

  attribute :model, :object
  attribute :trainer, :object
  attribute :train
  attribute :test


  after_initialize :set_device, :set_model, :set_optimizer
  after_initialize :separate_train_and_test, :set_iterators, :set_updater, :set_trainer

  def set_device
    @device = Chainer::Device.create(-1)
    Chainer::Device.change_default(@device)
  end

  def set_model
    self.model = Chainer::Links::Model::Classifier.new(ResNet18::Model.new(n_classes: class_label))
  end

  def separate_train_and_test
    if class_label == 10
      # 初回10分ほどかかる
      self.train, self.test = Chainer::Datasets::CIFAR.get_cifar10
    elsif class_label == 100
      self.train, self.test = Chainer::Datasets::CIFAR.get_cifar100
    end
  end

  def set_optimizer
    @optimizer = Chainer::Optimizers::MomentumSGD.new(lr: learning_rate)
    @optimizer.setup(model)
  end

  def set_iterators
    @train_iter = Chainer::Iterators::SerialIterator.new(train, batch_size)
    @test_iter = Chainer::Iterators::SerialIterator.new(test, batch_size, repeat: false, shuffle: false)
  end

  def set_updater
    @updater = Chainer::Training::StandardUpdater.new(@train_iter, @optimizer, device: @device)
  end

  def set_trainer
    self.trainer = Chainer::Training::Trainer.new(@updater, stop_trigger: [epoch, "epoch"], out: out)
  end

  def extend_trainer
    # modelの制度を評価する
    trainer.extend(EX::Evaluator.new(@test_iter, model, device: @device))
    #
    trainer.extend(EX::ExponentialShift.new('lr',0.5), trigger: [25, 'epoch'])
    # 学習中のモデルのスナップショット
    trainer.extend(EX::Snapshot.new,trigger: [epoch, "epoch"])
    #
    trainer.extend(EX::LogReport.new)
    trainer.extend(EX::PrintReport.new([
        "epoch", "main/loss", "validation/main/loss",
        "main/accuracy", "validation/main/accuracy", "elasped_time"
      ])
    )
    trainer.extend(EX::ProgressBar.new)
  end
end
