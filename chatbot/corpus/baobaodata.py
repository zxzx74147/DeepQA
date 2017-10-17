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

"""
Load data from a dataset of baobao data


"""

class BaobaoData:
    """
    """

    def __init__(self, baobaoFile):
        """
        Args:
            lightweightFile (string): file containing our lightweight-formatted corpus
        """
        self.conversations = []
        self.loadLines(baobaoFile )

    def loadLines(self, folderName):
        """
        Args:
            fileName (str): file to load
        """
        fileName = [f for f in listdir(folderName) if isfile(join(folderName, f))]

        filtrate = re.compile(u'[^\u4E00-\u9FA5A-Za-z0-9_\s]')
        emoji_pattern = re.compile(u"(\ud83d[\ude00-\ude4f])|"  # emoticons
                                   u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
                                   u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
                                   u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
                                   u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
                                   "+", flags=re.UNICODE)

        linesBuffer = []
        with open(folderName+os.sep+fileName[0], 'r', encoding='utf-8') as f:
            last_cid=0
            for line in f:
                temp = line.split('\t')
                if len(temp) < 4:
                    continue
                cid=int(temp[0])
                temp[3]=' '.join(jieba.cut(temp[3])).rstrip()
                temp[3] = filtrate.sub(r'', temp[3])  # 过滤掉标点符号
                temp[3] = emoji_pattern.sub(r'', temp[3])  # 过滤emoji
                if len(temp[3])<3:
                    continue
                if cid==last_cid:
                    linesBuffer.append({"time": temp[1],"uid": temp[2],"text": temp[3]})
                else:
                    if len(linesBuffer):
                        self.conversations.append({"cid":last_cid,"lines": linesBuffer})
                        linesBuffer = []
                    last_cid=cid;

            if len(linesBuffer):  # Eventually flush the last conversation
                self.conversations.append({"lines": linesBuffer})
                linesBuffer = []

    def getConversations(self):
        return self.conversations
