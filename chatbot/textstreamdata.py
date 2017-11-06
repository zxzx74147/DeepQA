# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
import collections
import  jieba
import subprocess
import tensorflow as tf

from chatbot.corpus.baobaowhisperdata import BaobaoDataWhisper
from chatbot.corpus.baobaowhisperfilterstreamdata import BaobaoWhisperFilterStreamData
from chatbot.corpus.baobaowhisperstreamdata import BaobaoWhisperStreamData
from chatbot.corpus.cornelldata import CornellData
from chatbot.corpus.opensubsdata import OpensubsData
from chatbot.corpus.scotusdata import ScotusData
from chatbot.corpus.ubuntudata import UbuntuData
from chatbot.corpus.lightweightdata import LightweightData
from chatbot.corpus.baobaodata import BaobaoData


class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []

class TextStreamData:
    """Dataset class
    Warning: No vocabulary limit
    """

    availableCorpus = collections.OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', CornellData),
        ('opensubs', OpensubsData),
        ('scotus', ScotusData),
        ('ubuntu', UbuntuData),
        ('lightweight', LightweightData),
        ('baobao',BaobaoData),
        ('baobaowhisper', BaobaoWhisperStreamData),
        ('baobaowhisperlite', BaobaoWhisperStreamData),
        ('baobaowhisperfilter', BaobaoWhisperFilterStreamData),
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextStreamData.availableCorpus.keys())

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Model parameters
        self.args = args
        self.count = 0
        # Path variables
        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        basePath = self._constructBasePath()
        self.fullSamplesPath = basePath + '.plk'  # Full sentences length/vocab
        self.fullSamplesDataPath = basePath + '.tfrecord'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.plk'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        )  # Sentences/vocab filtered for this model

        self.filteredSamplesDataPath = basePath + '-length{}-filter{}-vocabSize{}.tfrecord'.format(
            self.args.maxLength,
            self.args.filterVocab,
            self.args.vocabularySize,
        )  # Sentences/vocab filtered for this model

        self.padToken = -1  # Padding
        self.goToken = -1  # Start of sequence
        self.eosToken = -1  # End of sequence
        self.unknownToken = -1  # Word dropped from vocabulary
        self.GO = tf.constant([self.goToken])
        self.EOS = tf.constant([self.eosToken])

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.word2id = {}
        self.id2word = {}  # For a rapid conversion (Warning: If replace dict by list, modify the filtering to avoid linear complexity with del)
        self.idCount = {}  # Useful to filters the words (TODO: Could replace dict by list or use collections.Counter)

        self.loadCorpus()

        # Plot some stats:
        self._printStats()

        # if self.args.playDataset:
        #     self.playDataset()

    def _printStats(self):
        print('Loaded {}: {} words, {} QA'.format(self.args.corpus, len(self.word2id), self.count))

    def _constructBasePath(self):
        """Return the name of the base prefix of the current dataset
        """
        path = os.path.join(self.args.rootDir, 'data' + os.sep + 'samples' + os.sep)
        path += 'dataset-{}'.format(self.args.corpus)
        if self.args.datasetTag:
            path += '-' + self.args.datasetTag
        return path

    def makeLighter(self, ratioDataset):
        """Only keep a small fraction of the dataset, given by the ratio
        """
        #if not math.isclose(ratioDataset, 1.0):
        #    self.shuffle()  # Really ?
        #    print('WARNING: Ratio feature not implemented !!!')
        pass

    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.trainingSamples)



    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args.batchSize !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            sample = samples[i]
            if not self.args.test and self.args.watsonMode:  # Watson mode: invert question and answer
                sample = list(reversed(sample))
            if not self.args.test and self.args.autoEncode:  # Autoencode: use either the question or answer for both input and output
                k = random.randint(0, 1)
                sample = (sample[k], sample[k])
            # TODO: Why re-processed that at each epoch ? Could precompute that
            # once and reuse those every time. Is not the bottleneck so won't change
            # much ? and if preprocessing, should be compatible with autoEncode & cie.
            batch.encoderSeqs.append(list(reversed(sample[0])))  # Reverse inputs (and not outputs), little trick as defined on the original seq2seq paper
            batch.decoderSeqs.append([self.goToken] + sample[1] + [self.eosToken])  # Add the <go> and <eos> tokens
            batch.targetSeqs.append(batch.decoderSeqs[-1][1:])  # Same as decoder, but shifted to the left (ignore the <go>)

            # Long sentences should have been filtered during the dataset creation
            assert len(batch.encoderSeqs[i]) <= self.args.maxLengthEnco
            assert len(batch.decoderSeqs[i]) <= self.args.maxLengthDeco

            # TODO: Should use tf batch function to automatically add padding and batch samples
            # Add padding & define weight
            batch.encoderSeqs[i]   = [self.padToken] * (self.args.maxLengthEnco  - len(batch.encoderSeqs[i])) + batch.encoderSeqs[i]  # Left padding for the input
            batch.weights.append([1.0] * len(batch.targetSeqs[i]) + [0.0] * (self.args.maxLengthDeco - len(batch.targetSeqs[i])))
            batch.decoderSeqs[i] = batch.decoderSeqs[i] + [self.padToken] * (self.args.maxLengthDeco - len(batch.decoderSeqs[i]))
            batch.targetSeqs[i]  = batch.targetSeqs[i]  + [self.padToken] * (self.args.maxLengthDeco - len(batch.targetSeqs[i]))

        # Simple hack to reshape the batch
        encoderSeqsT = []  # Corrected orientation
        for i in range(self.args.maxLengthEnco):
            encoderSeqT = []
            for j in range(batchSize):
                encoderSeqT.append(batch.encoderSeqs[j][i])
            encoderSeqsT.append(encoderSeqT)
        batch.encoderSeqs = encoderSeqsT

        decoderSeqsT = []
        targetSeqsT = []
        weightsT = []
        for i in range(self.args.maxLengthDeco):
            decoderSeqT = []
            targetSeqT = []
            weightT = []
            for j in range(batchSize):
                decoderSeqT.append(batch.decoderSeqs[j][i])
                targetSeqT.append(batch.targetSeqs[j][i])
                weightT.append(batch.weights[j][i])
            decoderSeqsT.append(decoderSeqT)
            targetSeqsT.append(targetSeqT)
            weightsT.append(weightT)
        batch.decoderSeqs = decoderSeqsT
        batch.targetSeqs = targetSeqsT
        batch.weights = weightsT

        # # Debug
        # self.printBatch(batch)  # Input inverted, padding should be correct
        # print(self.sequence2str(samples[0][0]))
        # print(self.sequence2str(samples[0][1]))  # Check we did not modified the original sample

        return batch

    def getBatches(self):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        # filenames = tf.placeholder(tf.string, shape=[None])
        with tf.device('/cpu:0'):
            dataset = tf.contrib.data.TFRecordDataset([self.filteredSamplesDataPath])
            dataset = dataset.map(self.decodeQAExample)  # Parse the record into tensors.
            dataset = dataset.repeat()  # Repeat the input indefinitely.
            # dataset = dataset.batch(self.args.batchSize)

            dataset = dataset.padded_batch(1, padded_shapes=([self.args.maxLengthEnco],[self.args.maxLengthDeco],[self.args.maxLengthDeco],[self.args.maxLengthDeco]),
                                           padding_values=(self.padToken,self.padToken,self.padToken,0.0))
            dataset = dataset.shuffle(buffer_size=10000)
            iterator = dataset.make_initializable_iterator()

            next_batch = iterator.get_next()
            return iterator,next_batch
    def getSampleSize(self):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return self.count

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2id)

    def loadCorpus(self):
        """Load/create the conversations data
        """

        datasetExist = os.path.isfile(self.fullSamplesPath)  # Try to construct the dataset from the preprocessed entry
        if not datasetExist:
            print('Constructing full dataset...')

            optional = ''
            if self.args.corpus == 'lightweight':
                if not self.args.datasetTag:
                    raise ValueError('Use the --datasetTag to define the lightweight file to use.')
                optional = os.sep + self.args.datasetTag  # HACK: Forward the filename

            # Corpus creation
            corpusData = TextStreamData.availableCorpus[self.args.corpus](self.corpusDir + optional)
            self.createFullCorpus(corpusData.getConversations())
            self.saveDataset(self.fullSamplesPath)
        else:
            self.loadDataset(self.fullSamplesPath)

        self._printStats()
        datasetExist = os.path.isfile(self.filteredSamplesPath)
        if not datasetExist:
            self.filterFromFull()
            self.saveDataset(self.filteredSamplesPath)
        else:
            self.loadDataset(self.filteredSamplesPath)
        # datasetExist = os.path.isfile(self.filteredSamplesPath)
        # if not datasetExist:  # First time we load the database: creating all files
        #     print('Training samples not found. Creating dataset...')
        #
        #     datasetExist = os.path.isfile(self.fullSamplesPath)  # Try to construct the dataset from the preprocessed entry
        #     if not datasetExist:
        #         print('Constructing full dataset...')
        #
        #         optional = ''
        #         if self.args.corpus == 'lightweight':
        #             if not self.args.datasetTag:
        #                 raise ValueError('Use the --datasetTag to define the lightweight file to use.')
        #             optional = os.sep + self.args.datasetTag  # HACK: Forward the filename
        #
        #         # Corpus creation
        #         corpusData = TextSteamData.availableCorpus[self.args.corpus](self.corpusDir + optional)
        #         self.createFullCorpus(corpusData.getConversations())
        #         self.saveDataset(self.fullSamplesPath)
        #     else:
        #         self.loadDataset(self.fullSamplesPath)
        #     self._printStats()
        #
        #     print('Filtering words (vocabSize = {} and wordCount > {})...'.format(
        #         self.args.vocabularySize,
        #         self.args.filterVocab
        #     ))
        #     self.filterFromFull()  # Extract the sub vocabulary for the given maxLength and filterVocab
        #
        #     # Saving
        #     print('Saving dataset...')
        #     self.saveDataset(self.filteredSamplesPath)  # Saving tf samples
        # else:
            # self.loadDataset(self.filteredSamplesPath)


        # assert self.padToken == 0

    def saveDataset(self, filename):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """

        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2id': self.word2id,
                'id2word': self.id2word,
                'idCount': self.idCount,
                'count':self.count,
                # 'trainingSamples': self.trainingSamples
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2id = data['word2id']
            self.id2word = data['id2word']
            self.idCount = data.get('idCount', None)
            self.count = data['count']
            # self.trainingSamples = data['trainingSamples']

            self.padToken = self.word2id['<pad>']
            self.goToken = self.word2id['<go>']
            self.eosToken = self.word2id['<eos>']
            self.unknownToken = self.word2id['<unknown>']  # Restore special words
            self.GO = tf.constant([self.goToken])
            self.EOS = tf.constant([self.eosToken])


    def decodeQAExample(self,example_proto):
        features = {'Q': tf.VarLenFeature(tf.int64),
                    'A': tf.VarLenFeature(tf.int64),}
        parsed_features = tf.parse_single_example(example_proto, features)
        Q=tf.sparse_tensor_to_dense(parsed_features["Q"])
        A=tf.sparse_tensor_to_dense(parsed_features["A"])
        Q=tf.cast(Q, tf.int32)
        A=tf.cast(A, tf.int32)

        decoderSeq=tf.concat([self.GO,A,self.EOS],0)
        targetSeq=tf.concat([A,self.EOS],0)
        # weight = tf.ones_like(targetSeq)

        weight = tf.ones_like(targetSeq,dtype=tf.float32)
        # sum_w =tf.reduce_sum(weight)
        # weight = tf.scalar_mul(1.0 / sum_w, weight)
        # weight = tf.scalar_mul(tf.log(sum_w)/sum_w,weight)
        return Q,decoderSeq,targetSeq,weight


        # # encoderSeq = list(reversed(Q))
        # decoderSeq = tf.add([self.goToken],A)
        # decoderSeq = tf.add(decoderSeq,[self.eosToken])
        # # decoderSeq = [self.goToken] + A + [self.eosToken]
        # targetSeq=tf.add(A,[self.eosToken])
        # weight =targetSeq


        # encoderSeq=[self.padToken] * (self.args.maxLengthEnco  - len(encoderSeq))+encoderSeq
        # decoderSeq += [self.padToken] * (self.args.maxLengthDeco - len(decoderSeq))
        # weight = [1.0] * len(targetSeq) + [0.0] * (self.args.maxLengthDeco - len(targetSeq))
        # targetSeq = targetSeq+[self.padToken] * (self.args.maxLengthDeco - len(targetSeq))

        # return encoderSeq,decoderSeq,weight,targetSeq

    def decodeQA(self,filename):
        reader = tf.TFRecordReader()

        filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

        _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'Q': tf.VarLenFeature( tf.int64),
                                               'A': tf.VarLenFeature( tf.int64),
                                           })

        Q = tf.sparse_tensor_to_dense(features['Q'])  # 在流中抛出Q张量
        A = tf.sparse_tensor_to_dense(features['A'])  # 在流中抛出A张量
        return Q, A

    def filterFromFull(self):
        """ Load the pre-processed full corpus and filter the vocabulary / sentences
        to match the given model options
        """

        def mergeSentences(sentences, fromEnd=False):
            """Merge the sentences until the max sentence length is reached
            Also decrement id count for unused sentences.
            Args:
                sentences (list<list<int>>): the list of sentences for the current line
                fromEnd (bool): Define the question on the answer
            Return:
                list<int>: the list of the word ids of the sentence
            """
            # We add sentence by sentence until we reach the maximum length
            merged = []

            # If question: we only keep the last sentences
            # If answer: we only keep the first sentences
            if fromEnd:
                sentences = reversed(sentences)

            for sentence in sentences:

                # If the total length is not too big, we still can add one more sentence
                if len(merged) + len(sentence) <= self.args.maxLength:
                    if fromEnd:  # Append the sentence
                        merged = sentence + merged
                    else:
                        merged = merged + sentence
                else:  # If the sentence is not used, neither are the words
                    for w in sentence:
                        self.idCount[w] -= 1
            return merged


        dst = self.filteredSamplesDataPath+".temp"
        datasetExist = os.path.isfile(dst)
        if not datasetExist:
            Q, A = self.decodeQA(self.fullSamplesDataPath)
            with tf.Session() as sess,tf.python_io.TFRecordWriter(dst) as writer ,\
                    tqdm(total=self.count,desc='Filter') as pbar:  # 开始一个会话
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                count = 0
                for i in range(self.count):
                    pbar.update(1)
                    qustion, answer = sess.run([Q, A])  # 在会话中取出image和label
                    if len(qustion)>self.args.maxLengthEnco:
                        for w in qustion:
                            self.idCount[w] -= 1
                    if len(answer)>self.args.maxLengthEnco:
                        for w in answer:
                            self.idCount[w] -= 1

                     # Filter wrong samples (if one of the list is empty)
                    if qustion.size>self.args.maxLengthEnco or answer.size>self.args.maxLengthEnco:
                        continue

                    example = tf.train.Example(features=tf.train.Features(feature={
                            "Q": tf.train.Feature(int64_list=tf.train.Int64List(value=qustion)),
                            'A': tf.train.Feature(int64_list=tf.train.Int64List(value=answer))
                    }))  # example对象对label和image数据进行封装
                    writer.write(example.SerializeToString())
                    count=count+1


                    # print("Q", qustion)
                    # print("A", answer)
                self.count = count
                coord.request_stop()
                coord.join(threads)

        # newSamples = []
        self._printStats()


        # 1st step: Iterate over all words and add filters the sentences
        # according to the sentence lengths
        # for inputWords, targetWords in tqdm(self.trainingSamples, desc='Filter sentences:', leave=False):
        #     inputWords = mergeSentences(inputWords, fromEnd=True)
        #     targetWords = mergeSentences(targetWords, fromEnd=False)
        #
        #     newSamples.append([inputWords, targetWords])
        # words = []

        # WARNING: DO NOT FILTER THE UNKNOWN TOKEN !!! Only word which has count==0 ?

        # 2nd step: filter the unused words and replace them by the unknown token
        # This is also where we update the correnspondance dictionaries
        specialTokens = {  # TODO: bad HACK to filter the special tokens. Error prone if one day add new special tokens
            self.padToken,
            self.goToken,
            self.eosToken,
            self.unknownToken
        }
        newMapping = {}  # Map the full words ids to the new one (TODO: Should be a list)
        newId = 0

        selectedWordIds = collections \
            .Counter(self.idCount) \
            .most_common(self.args.vocabularySize or None)  # Keep all if vocabularySize == 0
        selectedWordIds = {k for k, v in selectedWordIds if v > self.args.filterVocab}
        selectedWordIds |= specialTokens

        for wordId, count in [(i, self.idCount[i]) for i in range(len(self.idCount))]:  # Iterate in order
            if wordId in selectedWordIds:  # Update the word id
                newMapping[wordId] = newId
                word = self.id2word[wordId]  # The new id has changed, update the dictionaries
                del self.id2word[wordId]  # Will be recreated if newId == wordId
                self.word2id[word] = newId
                self.id2word[newId] = word
                newId += 1
            else:  # Cadidate to filtering, map it to unknownToken (Warning: don't filter special token)
                newMapping[wordId] = self.unknownToken
                del self.word2id[self.id2word[wordId]]  # The word isn't used anymore
                del self.id2word[wordId]

        # Last step: replace old ids by new ones and filters empty sentences
        def replace_words(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] != self.unknownToken:  # Also filter if only contains unknown tokens
                    valid = True
            return valid

        def replace_words_t(words):
            valid = False  # Filter empty sequences
            for i, w in enumerate(words):
                words[i] = newMapping[w]
                if words[i] == self.unknownToken:  # Also filter if only contains unknown tokens
                    valid = False
                    return valid
                else:
                    valid= True
            return valid

        self.trainingSamples.clear()


        datasetExist = os.path.isfile(self.filteredSamplesDataPath)
        if not datasetExist:
            Q, A = self.decodeQA(dst)
            with tf.Session() as sess, tf.python_io.TFRecordWriter(self.filteredSamplesDataPath) as writer, \
                    tqdm(total=self.count, desc='Filter 2') as pbar:  # 开始一个会话
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                count = 0
                for i in range(self.count):
                    pbar.update(1)
                    qustion, answer = sess.run([Q, A])  # 在会话中取出image和label
                    valid = True
                    valid &= replace_words(qustion)
                    valid &= replace_words_t(answer)
                    # valid &= answer.count(self.unknownToken) == 0
                    if valid:
                        example = tf.train.Example(features=tf.train.Features(feature={
                            "Q": tf.train.Feature(int64_list=tf.train.Int64List(value=qustion)),
                            'A': tf.train.Feature(int64_list=tf.train.Int64List(value=answer))
                        }))  # example对象对label和image数据进行封装
                        # if count < 100:
                        #     tqdm.write("Q:" + self.sequence2str(qustion.tolist()))
                        #     tqdm.write("A:" + self.sequence2str(answer.tolist()))
                        writer.write(example.SerializeToString())
                        count = count + 1

                self.count = count
                coord.request_stop()
                coord.join(threads)

        # for inputWords, targetWords in tqdm(newSamples, desc='Replace ids:', leave=False):
        #     valid = True
        #     valid &= replace_words(inputWords)
        #     valid &= replace_words(targetWords)
        #     valid &= targetWords.count(self.unknownToken) == 0  # Filter target with out-of-vocabulary target words ?
        #
        #     if valid:
        #         self.trainingSamples.append([inputWords, targetWords])  # TODO: Could replace list by tuple

        self.idCount.clear()  # Not usefull anymore. Free data

    def createFullCorpus(self, conversations):
        """Extract all data from the given vocabulary.
        Save the data on disk. Note that the entire corpus is pre-processed
        without restriction on the sentence length or vocab size.
        """
        # Add standard tokens
        self.padToken = self.getWordId('<pad>')  # Padding (Warning: first things to add > id=0 !!)
        self.goToken = self.getWordId('<go>')  # Start of sequence
        self.eosToken = self.getWordId('<eos>')  # End of sequence
        self.unknownToken = self.getWordId('<unknown>')  # Word dropped from vocabulary

        count = self.linecount(conversations)

        # Preprocessing data

        with open(conversations, 'r', encoding='utf-8') as f,\
                tf.python_io.TFRecordWriter(self.fullSamplesDataPath) as writer ,\
                tqdm(total=count, desc='Conversation',leave=False) as pbar:
            inputWords = None
            targetWords = None
            index = 0
            for line in f:
                pbar.update(1)
                index = index + 1
                if line.startswith("//") or len(line.strip())==0:
                    inputWords = None
                    targetWords = None
                    continue
                if index % 2 == 1:
                    inputWords = self.extractText(line)
                    targetWords = None
                else:
                    targetWords = self.extractText(line)
                    if inputWords and targetWords:  # Filter wrong samples (if one of the list is empty)
                        example = tf.train.Example(features=tf.train.Features(feature={
                            "Q": tf.train.Feature(int64_list=tf.train.Int64List(value=inputWords)),
                            'A': tf.train.Feature(int64_list=tf.train.Int64List(value=targetWords))
                        }))  # example对象对label和image数据进行封装
                        # if index<100:
                        #     tqdm.write("Q:"+self.sequence2str(inputWords))
                        #     tqdm.write("A:" + self.sequence2str(targetWords))
                        writer.write(example.SerializeToString())
                        self.count=self.count+1
                        inputWords = None
                        targetWords = None




        # The dataset will be saved in the same order it has been extracted


    def linecount(self,path):
        count = int(subprocess.check_output(["wc",path]).split()[0])
        return count


    def extractText(self, line):
        """Extract the words from a sample lines
        Args:
            line (str): a line containing the text to extract
        Return:
            list<list<int>>: the list of sentences of word ids of the sentence
        """
        sentences = []  # List[List[str]]

        # Extract sentences
        sentencesToken = nltk.sent_tokenize(line)

        # We add sentence by sentence until we reach the maximum length
        for i in range(len(sentencesToken)):
            tokens = nltk.word_tokenize(sentencesToken[i])

            tempWords = []
            for token in tokens:
                tempWords.append(self.getWordId(token))  # Create the vocabulary and the training sentences

            sentences.append(tempWords)
        if (len(sentences)>0):
            return sentences[0]
        return None

    def getWordId(self, word, create=True):
        """Get the id of the word (and add it to the dictionary if not existing). If the word does not exist and
        create is set to False, the function will return the unknownToken value
        Args:
            word (str): word to add
            create (Bool): if True and the word does not exist already, the world will be added
        Return:
            int: the id of the word created
        """
        # Should we Keep only words with more than one occurrence ?

        word = word.lower()  # Ignore case

        # At inference, we simply look up for the word
        if not create:
            wordId = self.word2id.get(word, self.unknownToken)
        # Get the id if the word already exist
        elif word in self.word2id:
            wordId = self.word2id[word]
            self.idCount[wordId] += 1
        # If not, we create a new entry
        else:
            wordId = len(self.word2id)
            self.word2id[word] = wordId
            self.id2word[wordId] = word
            self.idCount[wordId] = 1

        return wordId

    def printBatch(self, batch):
        """Print a complete batch, useful for debugging
        Args:
            batch (Batch): a batch object
        """
        print('----- Print batch -----')
        for i in range(len(batch.encoderSeqs[0])):  # Batch size
            print('Encoder: {}'.format(self.batchSeq2str(batch.encoderSeqs, seqId=i)))
            print('Decoder: {}'.format(self.batchSeq2str(batch.decoderSeqs, seqId=i)))
            print('Targets: {}'.format(self.batchSeq2str(batch.targetSeqs, seqId=i)))
            print('Weights: {}'.format(' '.join([str(weight) for weight in [batchWeight[i] for batchWeight in batch.weights]])))

    def sequence2str(self, sequence, clean=False, reverse=False):
        """Convert a list of integer into a human readable string
        Args:
            sequence (list<int>): the sentence to print
            clean (Bool): if set, remove the <go>, <pad> and <eos> tokens
            reverse (Bool): for the input, option to restore the standard order
        Return:
            str: the sentence
        """

        if not sequence:
            return ''

        if not clean:
            return ' '.join([self.id2word[idx] for idx in sequence])

        sentence = []
        for wordId in sequence:
            if wordId == self.eosToken:  # End of generated sentence
                break
            elif wordId != self.padToken and wordId != self.goToken:
                sentence.append(self.id2word[wordId])

        if reverse:  # Reverse means input so no <eos> (otherwise pb with previous early stop)
            sentence.reverse()

        return self.detokenize(sentence)

    def detokenize(self, tokens):
        """Slightly cleaner version of joining with spaces.
        Args:
            tokens (list<string>): the sentence to print
        Return:
            str: the sentence
        """
        return ''.join([
            ' ' + t if not t.startswith('\'') and
                       t not in string.punctuation
                    else t
            for t in tokens]).strip().capitalize()

    def batchSeq2str(self, batchSeq, seqId=0, **kwargs):
        """Convert a list of integer into a human readable string.
        The difference between the previous function is that on a batch object, the values have been reorganized as
        batch instead of sentence.
        Args:
            batchSeq (list<list<int>>): the sentence(s) to print
            seqId (int): the position of the sequence inside the batch
            kwargs: the formatting options( See sequence2str() )
        Return:
            str: the sentence
        """
        sequence = []
        for i in range(len(batchSeq)):  # Sequence length
            sequence.append(batchSeq[i][seqId])
        return self.sequence2str(sequence, **kwargs)

    def sentence2enco(self, sentence):
        """Encode a sequence and return a batch as an input for the model
        Return:
            Batch: a batch object containing the sentence, or none if something went wrong
        """

        if sentence == '':
            return None

        sentence = ' '.join(jieba.cut(sentence))
        # First step: Divide the sentence in token
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) > self.args.maxLength:
            return None

        # Second step: Convert the token in word ids
        wordIds = []
        for token in tokens:
            wordIds.append(self.getWordId(token, create=False))  # Create the vocabulary and the training sentences

        # Third step: creating the batch (add padding, reverse)
        batch = self._createBatch([[wordIds, []]])  # Mono batch, no target output

        return batch

    def deco2sentence(self, decoderOutputs):
        """Decode the output of the decoder and return a human friendly sentence
        decoderOutputs (list<np.array>):
        """
        sequence = []

        # Choose the words with the highest prediction score
        for out in decoderOutputs:
            sequence.append(np.argmax(out))  # Adding each predicted word ids

        return sequence  # We return the raw sentence. Let the caller do some cleaning eventually

    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        for i in range(self.args.playDataset):
            idSample = random.randint(0, len(self.trainingSamples) - 1)
            print('Q: {}'.format(self.sequence2str(self.trainingSamples[idSample][0], clean=True)))
            print('A: {}'.format(self.sequence2str(self.trainingSamples[idSample][1], clean=True)))
            print()
        pass


def tqdm_wrap(iterable, *args, **kwargs):
    """Forward an iterable eventually wrapped around a tqdm decorator
    The iterable is only wrapped if the iterable contains enough elements
    Args:
        iterable (list): An iterable object which define the __len__ method
        *args, **kwargs: the tqdm parameters
    Return:
        iter: The iterable eventually decorated
    """
    if len(iterable) > 100:
        return tqdm(iterable, *args, **kwargs)
    return iterable
