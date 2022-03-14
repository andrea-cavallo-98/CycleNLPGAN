"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import logging
from util.visualizer import Visualizer
import time
import os
import gc
from tqdm import tqdm

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import torch

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s: %(message)s')
    opt = TrainOptions().parse()   # get training options

    torch.cuda.empty_cache()
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    #train_dataset, eval_dataset, test_dataset = create_dataset(opt, model)  # create a dataset given opt.dataset_mode and other options
    train_dataset, eval_dataset = create_dataset(opt, model)
    dataset_size = len(train_dataset)    # get the number of images in the dataset.
    logging.info('The number of training sentences = %d' % dataset_size)
    logging.info('The number of evaluation sentences = %d' % len(eval_dataset))
   # logging.info('The number of test sentences = %d' % len(test_dataset))
    logging.info('The number of training batches = %d' % len(train_dataset.dataloader))
    logging.info('The number of evaluation batches = %d' % len(eval_dataset.dataloader))
    #logging.info('The number of test batches = %d' % len(test_dataset.dataloader))

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = opt.iter_count                # the total number of training iterations


    n = round(opt.iter_count/opt.batch_size) #NBatch totali
    n -= (opt.epoch_count-1)*(round(len(train_dataset)/opt.batch_size))
    previous_suffix = None



    if not opt.continue_train:

        for j, eval_data in enumerate(eval_dataset.dataloader):  # inner loop within one epoch
            if j > 20:
                break
            model.set_input(eval_data)  # unpack data from dataset and apply preprocessing
            model.evaluate(sentences_file=os.path.join(opt.checkpoints_dir, opt.name, "0_0_sentence.txt"), 
                            sacre_file=os.path.join(opt.checkpoints_dir, opt.name, "0_0_sacre.tsv"))
            gc.collect()


        with open(os.path.join(opt.checkpoints_dir, opt.name, "0_0_sacre.tsv"), "r", encoding='utf8') as sacre_file:
            lines = sacre_file.read().split("\n")
            avg_fake_A = [float(e.split("\t")[0]) for e in lines if e != ""]
            avg_rec_A = [float(e.split("\t")[1]) for e in lines if e != ""]
            avg_fake_B = [float(e.split("\t")[2]) for e in lines if e != ""]
            avg_rec_B = [float(e.split("\t")[3]) for e in lines if e != ""]
            avg_fake_A = sum(avg_fake_A)/len(avg_fake_A)
            avg_rec_A = sum(avg_rec_A)/len(avg_rec_A)
            avg_fake_B = sum(avg_fake_B)/len(avg_fake_B)
            avg_rec_B = sum(avg_rec_B)/len(avg_rec_B)
        logging.info("BLEU real-fake A:" + str(avg_fake_A))
        logging.info("BLEU distance real-rec A:" + str(avg_rec_A))
        logging.info("BLEU distance real-fake B:" + str(avg_fake_B))
        logging.info("BLEU distance real-rec B:" + str(avg_rec_B))
        fw = open(os.path.join(opt.checkpoints_dir, opt.name, "average_sacre.tsv"), "a", encoding='utf8')
        fw.write("0\t0\t" + str(avg_fake_A) + "\t" + str(avg_rec_A) + "\t" + str(avg_fake_B) + "\t" + str(avg_rec_B) + "\n")
        fw.close()


    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        visualizer.print_current_lr(epoch, model.get_learning_rate())

        for i, data in enumerate(train_dataset.dataloader):  # inner loop within one epoch
            epoch_iter += opt.batch_size

            if epoch == opt.epoch_count:
                if n > 0:
                    n -= 1
                    continue
            iter_start_time = time.time()  # timer for computation per iteration

            model.set_input(data)         # unpack data from dataset and apply preprocessing

            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            total_iters += opt.batch_size

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)


            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logging.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'epoch_'+str(epoch)+'_iter_%d' % total_iters

                model.save_networks(save_suffix)
                if previous_suffix is not None:
                    model.delete_networks(previous_suffix)
                previous_suffix = save_suffix
                model.save_networks('latest')

            if opt.eval_freq is not None and total_iters % opt.eval_freq == 0:
                sentences_filename = str(epoch)+"_"+str(total_iters)+"_eval_sentences.txt"
                sacre_filename = str(epoch)+"_"+str(total_iters)+"_sacre.tsv"
                sentences_filename = os.path.join(opt.checkpoints_dir, opt.name, sentences_filename)
                sacre_filename = os.path.join(opt.checkpoints_dir, opt.name, sacre_filename)

                fw = open(sacre_filename, "w", encoding='utf8')
                fw.close()

                for j, eval_data in enumerate(eval_dataset.dataloader):  # inner loop within one epoch
                    if j > 20:
                        break
                    model.set_input(eval_data)  # unpack data from dataset and apply preprocessing
                    model.evaluate(sentences_file=sentences_filename, sacre_file=sacre_filename)


                with open(sacre_filename, "r", encoding='utf8') as sacre_file:
                    lines = sacre_file.read().split("\n")
                    avg_fake_A = [float(e.split("\t")[0]) for e in lines if e != ""]
                    avg_rec_A = [float(e.split("\t")[1]) for e in lines if e != ""]
                    avg_fake_B = [float(e.split("\t")[2]) for e in lines if e != ""]
                    avg_rec_B = [float(e.split("\t")[3]) for e in lines if e != ""]
                    avg_fake_A = sum(avg_fake_A) / len(avg_fake_A)
                    avg_rec_A = sum(avg_rec_A) / len(avg_rec_A)
                    avg_fake_B = sum(avg_fake_B) / len(avg_fake_B)
                    avg_rec_B = sum(avg_rec_B) / len(avg_rec_B)
                logging.info("BLEU real-fake A:" + str(avg_fake_A))
                logging.info("BLEU distance real-rec A:" + str(avg_rec_A))
                logging.info("BLEU distance real-fake B:" + str(avg_fake_B))
                logging.info("BLEU distance real-rec B:" + str(avg_rec_B))
                fw = open(os.path.join(opt.checkpoints_dir, opt.name, "average_sacre.tsv"), "a", encoding='utf8')
                fw.write(str(epoch)+"\t"+str(total_iters)+"\t" + str(avg_fake_A) + "\t" + str(avg_rec_A) + "\t" + str(avg_fake_B) + "\t" + str(avg_rec_B) + "\n")
                fw.close()

            iter_data_time = time.time()

        logging.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

        sacre_filename = os.path.join(opt.checkpoints_dir, opt.name, "sacre.tsv")

        with open(sacre_filename, "a", encoding='utf8') as sacre_file:
            sacre_file.write("NEW EPOCH:\n")

        for j, eval_data in enumerate(eval_dataset.dataloader):  # inner loop within one epoch
            model.set_input(eval_data)  # unpack data from dataset and apply preprocessing
            model.evaluate(sentences_file=sentences_filename, distance_file=distance_filename, mutual_avg_file=mutual_filename,  mutual_avg_file_A=mutual_filename_A, mutual_avg_file_B=mutual_filename_B, top_k_file=top_k_filename,sacre_file=sacre_filename)

        with open(sacre_filename, "a", encoding='utf8') as sacre_file:
            sacre_file.write("\n\n\n\n")

        logging.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
