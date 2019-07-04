class IrisChain < Chainer::Chain
  L = Chainer::Links::Connection::Linear
  F = Chainer::Functions

  def initialize(n_units, n_out)
    super()
    init_scope do
      @l1 = L.new(nil, out_size: n_units)
      @l2 = L.new(nil, out_size: n_out)
    end
  end

  def call(x, y)
    return F::Loss::MeanSquaredError.mean_squared_error(fwd(x),y)
  end

  def fwd(x)
    h1 = F::Activation::Sigmoid.sigmoid(@l1.(x))
    h2 = @l2.(h1)
    return h2
  end
end
