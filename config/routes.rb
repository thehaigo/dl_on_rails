Rails.application.routes.draw do
  resources :images do
    get :predict, on: :member
  end
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html
end
