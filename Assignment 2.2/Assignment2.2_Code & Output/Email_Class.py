# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 19:52:17 2016

@author: Yamuna
"""

class EmailTemplate:
    """A simple data object"""
    def __init__(self, filename, from_id, to_id, subject, body, wordcount):
        self.filename = filename
        self.from_id = from_id
        self.to_id = to_id
        self.subject = subject
        self.body = body
        self.wordcount = wordcount

    def __str__(self):
        return "EmailDisc {\n\tFilename: %s,\n\tFrom: %s,\n\tto_id %s,\n\tSubject: %s,\n\Body: %s,\n\WordCount: %s\n}" \
               % (self.filename, self.from_id, self.to_id, self.subject, self.body, self.wordcount)

    def __repr__(self):
        return "EmailDisc {\n\tFilename: %s,\n\tFrom: %s,\n\tto_id %s,\n\tSubject: %s,\n\Body: %s,\n\WordCount: %s\n}" \
               % (self.filename, self.from_id, self.to_id, self.subject, self.body, self.wordcount)