from model import *
from data_import import *
import sys, getopt
from scipy.io import wavfile
from scipy import signal
import os

class Denoising():
    """
    Denoising Class holds all the necessary functions for denoising the noisy samples.
    """

    def __init__(self, noisy_speech_folder='', sampled_noisy_speech_folder='', modfolder=''):

        self.modfolder = modfolder
        self.noisy_speech_folder = noisy_speech_folder
        self.sampled_noisy_speech_folder = sampled_noisy_speech_folder

    # SAMPLING FUNCTION
    def sampling(self):
        '''
        Converts the input noisy audio files into required format and samples it to 16kHz.
        
        '''

        fs = 16000
        filelist = os.listdir("%s"%(self.noisy_speech_folder))
        filelist = [f for f in filelist if f.endswith(".wav")]
        if not os.path.exists(self.sampled_noisy_speech_folder):
            os.makedirs(self.sampled_noisy_speech_folder)

        for i in tqdm(filelist):
            sr, y = wavfile.read("%s/%s" % (self.noisy_speech_folder, i))
            if y.dtype == 'int16':
                nb_bits = 16 # -> 16-bit wav files
            elif y.dtype == 'int32':
                nb_bits = 32 # -> 32-bit wav files
            # converting to 32 point floating values
            y_float = y.astype(float) / (2.0**(nb_bits-1) + 1)
            # sampling to 16kHz
            samples = round(len(y_float) * fs/sr) # Number of samples to downsample
            Y = signal.resample(y_float, int(samples))
            wavfile.write(os.path.join(self.sampled_noisy_speech_folder, str(i)), fs, Y)

        print "Converted all the input noisy samples to required format. The corresponding sampled audio files are present in the specified folder."


    # INFERENCE FUNCTION
    def inference(self, SE_LAYERS = 13, SE_CHANNELS = 64, SE_NORM = "NM", fs = 16000):
        '''
        Denoises the noisy samples and produces the corresponding denoised samples in the specified path.
        Args:
            SE_LAYERS (int) : Number of Internal Layers of the SENET model
            SE_CHANNELS (int) : Number of feature channels per layer
            SE_NORM (string) : Type of layer normalization (NM, SBN or None)
            fs (int) : Sampling frequency or rate

        '''

        datafolder = self.sampled_noisy_speech_folder
        if datafolder[-1] == '/':
            datafolder = datafolder[:-1]
        if not os.path.exists(datafolder+'_denoised'):
            os.makedirs(datafolder+'_denoised')

        # LOAD DATA
        dataset = load_noisy_data_list(valfolder = datafolder)
        dataset = load_noisy_data(dataset)

        # SET LOSS FUNCTIONS AND PLACEHOLDERS
        with tf.variable_scope(tf.get_variable_scope()):
            input=tf.placeholder(tf.float32,shape=[None,1,None,1])
            clean=tf.placeholder(tf.float32,shape=[None,1,None,1])

            enhanced=senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)

        # INITIALIZE GPU CONFIG
        config=tf.ConfigProto()
        # config.gpu_options.allow_growth=True
        sess=tf.Session(config=config)
        print "Config ready"
        sess.run(tf.global_variables_initializer())
        print "Session initialized"
        saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])
        saver.restore(sess, "%s/se_model.ckpt" % self.modfolder)

        for id in tqdm(range(0, len(dataset["innames"]))):
            i = id # NON-RANDOMIZED ITERATION INDEX
            inputData = dataset["inaudio"][i] # LOAD DEGRADED INPUT
            # VALIDATION ITERATION
            output = sess.run([enhanced], feed_dict={input: inputData})
            output = np.reshape(output, -1)
            wavfile.write("%s_denoised/%s" % (datafolder,dataset["shortnames"][i]), fs, output)

        print "Denoised samples of the corresponding noisy samples have been created in the mentioned folder."





# MAIN
# if __name__ == '__main__':
#     noisy_speech_folder = 'datasets/noisy_speech'
#     sampled_noisy_speech_folder = 'datasets/sampled_noisy_speech'
#     modfolder = "models"
#     denoise = Denoising(noisy_speech_folder=noisy_speech_folder, sampled_noisy_speech_folder=sampled_noisy_speech_folder, modfolder=modfolder)
#     denoise.sampling()
#     denoise.inference()
    # datafolder = sampled_noisy_speech_folder
    # inference(valfolder=datafolder, modfolder=modfolder)
