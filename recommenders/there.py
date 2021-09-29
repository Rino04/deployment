def get_model(model):
    import keras
    from IPython.display import SVG
    from keras import optimizers
    from keras.utils.vis_utils import model_to_dot
    n_users, n_product = len(ratings_df.user_id.unique()), len(dataset.item_id.unique())
    n_latent_factors = 3

    n_latent_factors_user = 5
    n_latent_factors_product = 8

    product_input = keras.layers.Input(shape=[1],name='Item')
    product_embedding = keras.layers.Embedding(n_product + 1, n_latent_factors_product, name='product-Embedding')(product_input)
    product_vec = keras.layers.Flatten(name='Flattenproducts')(product_embedding)
    product_vec = keras.layers.Dropout(0.2)(product_vec)


    user_input = keras.layers.Input(shape=[1],name='User')
    user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input))
    user_vec = keras.layers.Dropout(0.2)(user_vec)


    concat = keras.layers.merge.concatenate([product_vec, user_vec],name='Concat')
    concat_dropout = keras.layers.Dropout(0.2)(concat)
    dense = keras.layers.Dense(200,name='FullyConnected')(concat)
    dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
    dense_2 = keras.layers.Dense(100,name='FullyConnected-1')(concat)
    dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
    dense_3 = keras.layers.Dense(50,name='FullyConnected-2')(dense_2)
    dropout_3 = keras.layers.Dropout(0.2,name='Dropout')(dense_3)
    dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)

    from keras import optimizers
    result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)

    model = keras.Model([user_input, product_input], result)
    model.compile(optimizer='adam',loss= 'mean_absolute_error')
    history = model.fit([train.user_id, train.item_id], train.rating, validation_split=0.1, epochs=25, verbose=0)