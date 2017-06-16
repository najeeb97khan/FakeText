## Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import time

## Importing utilities
import sys
sys.path.insert(0, '../utils')
from get_data import *
from make_batches import *
from encode_decode import *

## Defining Constants
HIDDEN_SIZE = 200
NUM_STEPS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
LEN_SENTENCE = 50
SKIP_STEP = 100
DATA_PATH = '../data/arvix_abstracts.txt'

## Constructing the GRU cell
def constructRNN(sequence, hidden_size, vocab, layer_name):
    
    with tf.variable_scope(layer_name) as scope:
        seq = tf.one_hot(sequence, len(vocab))
        cell = tf.contrib.rnn.GRUCell(hidden_size)
        in_state = tf.placeholder_with_default(cell.zero_state(tf.shape(seq)[0], tf.float32), shape=[None, hidden_size])
        length = tf.reduce_sum(tf.reduce_max(tf.sign(seq),2), 1)
        output, out_state = tf.nn.dynamic_rnn(cell, seq, length, in_state)
        return seq, output, out_state, in_state
    
def constructModel(vocab):
    
    with tf.variable_scope("placeholder") as scope:
        seq = tf.placeholder(tf.int32, shape=[None, None])
    
    seq_one_hot, output, out_state, in_state = constructRNN(seq, HIDDEN_SIZE, vocab, "GRU")
    
    with tf.variable_scope("fully_connected") as scope:
        logits = tf.contrib.layers.fully_connected(output, len(vocab), None)

    with tf.variable_scope("loss") as scope:
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:,:-1], labels=seq_one_hot[:,1:]))
    
    return seq, loss, output, out_state, in_state, logits

def createOptimizer(loss, global_step, name="optimizer"):
    with tf.variable_scope(name) as scope:
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
        return optimizer

def createSummaries(loss, loss_name):
    with tf.variable_scope("summaries") as scope:
        tf.summary.scalar(loss_name, loss)
        tf.summary.histogram(loss_name, loss)
        summary_op = tf.summary.merge_all()
        return summary_op

def train(seq, vocab, loss, optimizer, summary_op, in_state, out_state, logits, global_step):

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        ## Creating file writer
        writer = tf.summary.FileWriter('../graphs/arXiv', sess.graph)
        
        ## Creating checkpoint saver
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/arXiv/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            print 'Checkpoint Found. Restoring session...'
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        ## Training the model
        start = time.time()
        iteration = global_step.eval()
        for batch in read_batch(read_data(DATA_PATH, vocab)):
            batch_loss, summary, _ = sess.run([loss, summary_op, optimizer], feed_dict={seq:batch})

            if iteration % SKIP_STEP == 0:
                print 'Iteration:{}\n Loss: {}\tTime: {}'.format(iteration, batch_loss, time.time() - start)
                generate_text(sess, vocab, in_state, out_state, seq, logits, seed='T')
                start = time.time()
                saver.save(sess, '../checkpoints/arXiv/fake', iteration)
            iteration += 1
            writer.add_summary(summary, iteration)

def generate_text(sess, vocab, in_state, out_state, seq, logits, seed='T'):
    
    sentence = seed
    state = None
    for _ in range(LEN_SENTENCE):
        
        batch = [encode_vocab(sentence[-1], vocab)]
        feed = {seq:batch}
        if state is not None:
            feed.update({in_state:state})
        output, state = sess.run([logits, out_state], feed)
        inx = np.argmax(np.reshape(np.asarray(output),[1, len(vocab)]))
        sentence += decode_vocab([inx], vocab)
    
    print sentence
    return

def main():
    vocab = (" $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "\\^_abcdefghijklmnopqrstuvwxyz{|}")

    global_step = tf.Variable(0, tf.float32, name="global_step")
    print 'Constructing the model'
    seq, loss, output, out_state, in_state, logits = constructModel(vocab)
    optimizer = createOptimizer(loss, global_step)
    summary_op = createSummaries(loss, "xentropy_loss")
    print 'Training the model'
    train(seq, vocab, loss, optimizer, summary_op, in_state, out_state, logits, global_step)


if __name__ == "__main__":
    main()