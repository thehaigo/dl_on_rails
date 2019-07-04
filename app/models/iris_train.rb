class IrisTrain < ActiveType::Object

  attribute :x, :object
  attribute :y, :object
  attribute :y_onehot, :object
  attribute :model, :object
  after_initialize :set_device, :set_model, :set_optimizer
  after_initialize :set_dataset, :set_input_and_label, :separate_train_and_test

  def set_device
    # cpu mode
    device = Chainer::Device.create(-1)
    Chainer::Device.change_default(device)
    @device = device.xm
  end

  def set_model
    self.model = IrisChain.new(6,3)
  end

  def set_optimizer
    @optimizer = Chainer::Optimizers::Adam.new
    @optimizer.setup(model)
  end

  def set_dataset
    @dataset = Datasets::Iris.new
  end

  def set_input_and_label
    iris_table = @dataset.to_table

    x = iris_table.fetch_values(
      :sepal_length,
      :sepal_width,
      :petal_length,
      :petal_width
    ).transpose

    y_class = iris_table[:label]
    # class index array
    # ["Iris-setosa", "Iris-versicolor","Iris-virginica"]
    class_name = y_class.uniq
    # label名を数値に変換
    # y => [0,0,0,0, ,,,, 1,1, ,,,2,2]
    y = y_class.map{|s| class_name.index(s)}
    y_onehot = @device::SFloat.eye(class_name.size)[y, false]
    # NArrayに変換
    self.x = @device::SFloat.cast(x)
    self.y = @device::SFloat.cast(y)
    self.y_onehot = @device::SFloat.cast(y_onehot)
    true
  end

  def separate_train_and_test
    # 学習用に奇数行を抽出
    @x_train = x[(1..-1).step(2), true]
    @y_train = y_onehot[(1..-1).step(2), true]
    # 検証用に偶数行を抽出
    @x_test = x[(0..-1).step(2), true]
    @y_test = y[(0..-1).step(2)]
    true
  end

  def train
    1000.times do |i|
      print(".") if i % 1000 == 0
      x = Chainer::Variable.new(@x_train)
      y = Chainer::Variable.new(@y_train)
      model.cleargrads()
      loss = model.(x, y)
      loss.backward()
      @optimizer.update()
    end
  end

  def test
    xt = Chainer::Variable.new(@x_test)
    yt = model.fwd(xt)
    n_row, n_col = yt.data.shape
    puts "Result : Correct Answer : Answer <= One-Hot"
    ok = 0
    n_row.times{|i|
      ans = yt.data[i, true].max_index()
      if ans == @y_test[i]
        ok += 1
        printf("OK")
      else
        printf("--")
      end
      printf(" : #{@y_test[i].to_i} :")
      puts " #{ans.to_i} <= #{yt.data[i, 0..-1].to_a}"
    }
    puts "Row: #{n_row}, Column: #{n_col}"
    puts "Accuracy rate : #{ok}/#{n_row} = #{ok.to_f / n_row}"
  end

  def load
    Chainer::Serializers::MarshalDeserializer.load_file("./lib/weights/iris.weight", model)
  end

  def save
    Chainer::Serializers::MarshalSerializer.save_file("./lib/weights/iris.weight", model)
  end
end
