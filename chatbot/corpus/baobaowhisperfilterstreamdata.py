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

import os
import jieba
from os import listdir
from os.path import isfile, join
import re
import tensorflow as tf
import linecache
from tqdm import tqdm  # Progress bar
import subprocess
import time
"""
Load data from a dataset of baobao data


"""


class BaobaoWhisperFilterStreamData:
    """
    """

    def __init__(self, baobaoFile):
        """
        Args:
            lightweightFile (string): file containing our lightweight-formatted corpus
        """

        file = baobaoFile+os.sep + 'whisper_train.txt'
        self.conversations = self.loadLines(file)
        # file = baobaoFile + os.sep + 'whisper_dev.txt'
        # self.test_conversations = self.loadLines(file)

    def linecount(self,path):
        count = int(subprocess.check_output(["wc",path]).split()[0])
        return count

    def loadLines(self, folderName):
        """
        Args:
            fileName (str): file to load
        """

        src_file = folderName
        dst_file = folderName+'.chat'

        if os.path.isfile(dst_file):
            return dst_file
        count = self.linecount(src_file)
        print(count)
        filtrate = re.compile(u'[^\u4E00-\u9FA5A-Za-z0-9_\s]')
        emoji_pattern = re.compile(u"(\ud83d[\ude00-\ude4f])|"  # emoticons
                                   u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
                                   u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
                                   u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
                                   u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
                                   "+", flags=re.UNICODE)

        linesBuffer = []
        lastQ = ''
        print('Read baobao whisper data:')
        with open(src_file, 'r', encoding='utf-8') as f, open(dst_file, 'w', encoding='utf-8')as fo,tqdm(total=count,desc='PreProcess') as pbar:
            for line in f:
                pbar.update(1)
                if line.startswith("Q: "):
                    line = line.replace("Q: ",'')
                    line = ' '.join(jieba.cut(line)).rstrip()
                    line = filtrate.sub(r'', line)  # 过滤掉标点符号
                    line = emoji_pattern.sub(r'', line)  # 过滤emoji
                    if lastQ!=line:
                        lastQ = line
                    fo.write(line+'\n')
                if line.startswith("A: "):
                    line = line.replace("A: ", '')
                    line = ' '.join(jieba.cut(line)).rstrip()
                    line = filtrate.sub(r'', line)  # 过滤掉标点符号
                    line = emoji_pattern.sub(r'', line)  # 过滤emoji
                    fo.write(line+'\n')
                    # fo.write("//new_chat" + '\n')
        return dst_file
    def getConversations(self):
        print('path='+self.conversations)
        return self.conversations

    def getTestConversations(self):
        print('path='+self.test_conversations)
        return self.test_conversations
