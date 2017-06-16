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
HIDDEN_SIZE = 5
NUM_STEPS = 60
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
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
    
    return seq, loss, output, out_state, in_state

def createOptimizer(loss, global_step, name="optimizer"):
    with tf.variable_scope(name) as scope:
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)
        return optimizer

def train(seq, vocab, loss, optimizer, global_step):

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
            batch_loss = sess.run([loss, optimizer], feed_dict={seq:batch})

            if iteration % SKIP_STEP:
                print 'Iteration:{}\n Loss: {}\tTime: {}'.format(iteration, batch_loss, time.time() - start)
                generate_text(sess, vocab, seed='T')
                start = time.time()
                saver.save(sess, '../checkpoints/arXiv/fake', iteration)
            iteration += 1

def generate_text(sess, vocab, seed='T'):
    
    print 'Text generation function not implemented...'
    return

def main():
    vocab = (" $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
         "\\^_abcdefghijklmnopqrstuvwxyz{|}")

    global_step = tf.Variable(0, tf.float32, name="global_step")
    print 'Constructing the model'
    seq, loss, output, out_state, in_state = constructModel(vocab)
    optimizer = createOptimizer(loss, global_step)
    print 'Training the model'
    train(seq, vocab, loss, optimizer, global_step)


if __name__ == "__main__":
    main()