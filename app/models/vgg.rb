class Vgg < ActiveType::Object
  CONV1_1_IN_NAME = 'Input_0'.freeze
  FC6_OUT_NAME = 'Gemm_0'.freeze
  SOFTMAX_OUT_NAME = 'Softmax_0'.freeze
  attribute :onnx, :object
  attribute :model, :object
  attribute :image_list, :object
  attribute :results

  after_initialize :set_onnx, :set_model

  def set_onnx
    self.onnx = Menoh::Menoh.new("./lib/model/VGG16.onnx")
  end

  def set_model
    self.model = onnx.make_model(model_opt)
  end

  def predict
    self.results = sort_result(model.run(image_set))

  end

  def image_set
    [
      name: CONV1_1_IN_NAME,
      data: image_list.map do |img|
        image = Magick::Image.read(img.file.path).first
        image = image.resize_to_fill(224,224)
        "RGB".split("").map do |color|
          image.export_pixels(0, 0, image.columns, image.rows, color).map do |pixel|
            pixel / 256 - rgb_offset[color.to_sym]
          end
        end.flatten
      end.flatten
    ]
  end

  def model_opt
    {
      backend: 'mkldnn',
      input_layers: [
        {
          name: CONV1_1_IN_NAME,
          dims: [1,3,224,224]
        }
      ],
      output_layers: [FC6_OUT_NAME, SOFTMAX_OUT_NAME]
    }
  end

  def rgb_offset
    {
      R: 123.68,
      G: 116.779,
      B: 103.939
    }
  end

  def sort_result(results)
    categories = File.read('./lib/data/synset_words.txt').split("\n")
    layer_result = results.find { |x| x[:name] == SOFTMAX_OUT_NAME }
    layer_result[:data].zip(image_list).each do |image_result, image_filepath|
      # sort by score
      sorted_result = image_result.zip(categories).sort_by { |x| -x[0] }
      return sorted_result[0,2]
    end
  end
end
