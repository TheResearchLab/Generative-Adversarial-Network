import argparse
import os 
import pandas as pd
import json

#import keras
#from keras.models import Sequential
#from keras.layers import Reshape
#from keras.layers import Flatten
#from keras.layers import Conv2D, Dense, Conv2DTranspose
#from keras.layers import Dropout
#from keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import tensorflow as tf
import boto3
from boto3 import Session

session = Session()
credentials = session.get_credentials()

#aws access credentials
current_credentials = credentials.get_frozen_credentials()





def model(args, x_train,client):
    
    adam = Adam(learning_rate=0.0002)
    # Function for Generator
    def build_generator():
        model = tf.keras.models.Sequential()

        # Layer 1
        model.add(tf.keras.layers.Dense(256 * 16* 16, input_dim=args.latent_dim)) # 128/2/2/2 because 3 conv2d layers?
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Reshape((16,16,256)))

        # Layer 2
        model.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        # Layer 3
        model.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        # Layer 4
        model.add(tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        #Layer 5
        model.add(tf.keras.layers.Conv2D(3, (3,3), activation='tanh', padding='same'))

        #model.summary()

        return model

    # Function for Discriminator
    
    
    def build_discriminator():
        model = tf.keras.models.Sequential()

        # Layer 1
        model.add(tf.keras.layers.Conv2D(64, (3,3), padding='same', input_shape=(args.image_dim,args.image_dim,3)))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        # Layer 2
        model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same', ))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        # Layer 3
        model.add(tf.keras.layers.Conv2D(128, (3,3), padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        # Layer 4
        model.add(tf.keras.layers.Conv2D(256, (3,3), padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

        # Layer 5
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.4)) # Drop 40% of neurons
        model.add(tf.keras.layers.Dense(1, activation='sigmoid')) 

        #model.summary()
        return model



    def train(x_train,epochs, batch_size=64):

        
        #Rescale data between -1 and 1
        x_train = x_train / 255 #/ 127.5 -1.
        bat_per_epo = int(x_train.shape[0] / batch_size)
        x_train = x_train.reshape(x_train.shape[0],args.image_dim,args.image_dim,3) 

        #Create our Y for our Neural Networks
        valid = np.ones((batch_size, 1))
        fakes = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            for j in range(bat_per_epo):
                #Get Random Batch
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                print(idx)
                imgs = x_train[idx]
                #Generate Fake Images
                noise = np.random.normal(0, 1, (batch_size, args.latent_dim)) #100 for latent_dim
                gen_imgs = generator.predict(noise)

                #Train discriminator
                d_loss_real = discriminator.train_on_batch(imgs, valid)
                d_loss_fake = discriminator.train_on_batch(gen_imgs, fakes)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                noise = np.random.normal(0, 1, (args.batch_size, args.latent_dim))

                #inverse y label
                g_loss = GAN.train_on_batch(noise, valid)
                
                print("******* %d %d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch,j, d_loss[0], 100* d_loss[1], g_loss))
                
            #Save the latest model
            
            generator.save('my_generator.h5') 
            client.upload_file(Filename='my_generator.h5',Bucket='tellisa-collage-gan',Key='my_generator.h5')
            
            discriminator.save('my_discriminator.h5')
            client.upload_file(Filename='my_discriminator.h5',Bucket='tellisa-collage-gan',Key='my_discriminator.h5')
        
        



    # GAN Setup 1
    generator = build_generator() #Build generator
    discriminator = build_discriminator() # Build discriminator
    discriminator.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        
    #GAN Setup 2
    # Combine into one model
    GAN = tf.keras.models.Sequential()
    discriminator.trainable = False #discriminator weights don't train with GAN
    GAN.add(generator)
    GAN.add(discriminator)

    GAN.compile(loss='binary_crossentropy', optimizer=adam)
        
        
    train(x_train, args.epochs,batch_size=args.batch_size)
        
    return generator #ouput discriminator too!

def _load_data(file_path, channel):
    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(file_path, file) for file in os.listdir(file_path) ]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(file_path, channel))
        
    raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
    df = pd.concat(raw_data)  
    
    features = df.values #df.iloc[:,:-1].values
    return features



def _parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters are described here.
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=100)
    parser.add_argument('--image_dim',type=int,default=128)
    

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TESTING'))
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    
    client = boto3.client(
    's3',
    aws_access_key_id=current_credentials.access_key,
    aws_secret_access_key=current_credentials.secret_key,
    aws_session_token=current_credentials.token)

    train_data = _load_data(args.train,'train')

    gan = model(args,train_data,client)

    if args.current_host == args.hosts[0]:
        # save model to an S3 directory with version number '00000001'
        gan.save(os.path.join(args.sm_model_dir, '000000001'), 'my_model.h5')