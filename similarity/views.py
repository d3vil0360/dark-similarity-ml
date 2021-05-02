from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

similarity_model = settings.SIMILARITY_MODEL
graph = tf.get_default_graph()

@api_view(['POST'])
def similarity(req):
    MAX_NB_WORDS = 200000
    MAX_SEQUENCE_LENGTH = 255
    questions = [req.data['q1'], req.data['q2']]
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(questions)

    question1_word_sequences = tokenizer.texts_to_sequences([questions[0]])
    question2_word_sequences = tokenizer.texts_to_sequences([questions[1]])
    q1_data = pad_sequences(question1_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    q2_data = pad_sequences(question2_word_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    global graph
    with graph.as_default():
        out = similarity_model.predict([np.array(q1_data), np.array(q2_data)], steps=None, verbose=0)
        api_overview = {
            'success' : 1,
            'result': out[0][0]
        }
    
    return Response(api_overview)

