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
from comet_ml import Experiment

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s: %(message)s')
    opt = TrainOptions().parse()   # get training options

    torch.cuda.empty_cache()
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    
    ''' Setting up Comet if requested '''
    if opt.comet_key is not None:
    
        # Create an experiment with your api key
        experiment = Experiment(
            api_key=opt.comet_key,
            project_name=opt.comet_exp_name,
            workspace=opt.comet_workspace,
        )
        
        hyper_params = {
            "learning_rate": opt.lr,
            "batch_size": opt.batch_size,
        }
        experiment.log_parameters(hyper_params)
    else:
        experiment = None
    

    (train_dataset_A_mono, train_dataset_B_mono, eval_dataset_A_mono, eval_dataset_B_mono), \
        (train_dataset_bi, eval_dataset_bi) = create_dataset(opt)
    dataset_size = len(train_dataset_A_mono)    # get the number of images in the dataset.
    logging.info('The number of training sentences = %d' % dataset_size)
    logging.info('The number of evaluation sentences = %d' % len(eval_dataset_A_mono))
    logging.info('The number of evaluation sentences = %d' % len(eval_dataset_B_mono))
    logging.info('The number of training batches = %d' % len(train_dataset_A_mono.dataloader))
    logging.info('The number of evaluation batches = %d' % len(eval_dataset_A_mono.dataloader))

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = opt.iter_count                # the total number of training iterations

    n = len(train_dataset_bi.dataloader)
    previous_suffix = None

    print("\n\n ####### START TRAINING ######### \n\n")

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        visualizer.print_current_lr(epoch, model.get_learning_rate())

        train_dataset_A_mono_iter = enumerate(train_dataset_A_mono.dataloader)
        train_dataset_B_mono_iter = enumerate(train_dataset_B_mono.dataloader)
        train_dataset_bi_iter = enumerate(train_dataset_bi.dataloader)

        print("****** Epoch ", epoch, " *******")
        for i in tqdm(range(n)):  # inner loop within one epoch
            epoch_iter += opt.batch_size

            ###
            # Perform one SUPERVISED iteration
            ###
            print("--- SUPERVISED iteration")

            data_A, data_B = train_dataset_bi_iter.__next__()
            model.set_input(data_A, data_B)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters_bilingual()   # calculate loss functions, get gradients, update network weights

            ###
            # Perform the UNSUPERVISED iterations
            ###
            print("--- UNSUPERVISED iteration")
            
            for it in tqdm(range(opt.ratio)):
                _, data_A = train_dataset_A_mono_iter.__next__()
                _, data_B = train_dataset_B_mono_iter.__next__()

                iter_start_time = time.time()  # timer for computation per iteration

                model.set_input(data_A, data_B)         # unpack data from dataset and apply preprocessing
                model.optimize_parameters_monolingual()   # calculate loss functions, get gradients, update network weights


            total_iters += opt.batch_size

            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if experiment is not None:
                    for k, v in losses.items():
                        experiment.log_metric(f"{k}:", v, step=total_iters)

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
                sentences_filename = os.path.join(opt.checkpoints_dir, opt.name, sentences_filename)
                
                eval_dataset_A_iter = enumerate(eval_dataset_A_mono.dataloader)
                eval_dataset_B_iter = enumerate(eval_dataset_B_mono.dataloader)
                
                for j in tqdm(range(5)):  # inner loop within one epoch
                    _, eval_data_A = eval_dataset_A_iter.__next__()
                    _, eval_data_B = eval_dataset_B_iter.__next__()
                    model.set_input(eval_data_A, eval_data_B)  # unpack data from dataset and apply preprocessing
                    model.evaluate_unsupervised(sentences_file=sentences_filename)
                
                
            '''
            if opt.eval_freq is not None and total_iters % opt.eval_freq == 0:
                sentences_filename = str(epoch)+"_"+str(total_iters)+"_eval_sentences.txt"
                sacre_filename = str(epoch)+"_"+str(total_iters)+"_sacre.tsv"
                sentences_filename = os.path.join(opt.checkpoints_dir, opt.name, sentences_filename)
                sacre_filename = os.path.join(opt.checkpoints_dir, opt.name, sacre_filename)

                fw = open(sacre_filename, "w", encoding='utf8')
                fw.close()

                eval_dataset_A_iter = enumerate(eval_dataset_A.dataloader)
                eval_dataset_B_iter = enumerate(eval_dataset_B.dataloader)

                for j in range(20):  # inner loop within one epoch
                    _, eval_data_A = eval_dataset_A_iter.__next__()
                    _, eval_data_B = eval_dataset_B_iter.__next__()
                    model.set_input(eval_data_A, eval_data_B)  # unpack data from dataset and apply preprocessing
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
                
                if experiment is not None:
                    experiment.log_metric("BLEU real-fake A:", avg_fake_A, step=0)
                    experiment.log_metric("BLEU distance real-rec A:", avg_rec_A, step=0)
                    experiment.log_metric("BLEU distance real-fake B:", avg_fake_B, step=0)
                    experiment.log_metric("BLEU distance real-rec B:", avg_rec_B, step=0)
                    
                fw = open(os.path.join(opt.checkpoints_dir, opt.name, "average_sacre.tsv"), "a", encoding='utf8')
                fw.write(str(epoch)+"\t"+str(total_iters)+"\t" + str(avg_fake_A) + "\t" + str(avg_rec_A) + "\t" + str(avg_fake_B) + "\t" + str(avg_rec_B) + "\n")
                fw.close()
            '''
            
                

            iter_data_time = time.time()

        logging.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
        model.save_networks('latest')
        model.save_networks(epoch)

        sacre_filename = os.path.join(opt.checkpoints_dir, opt.name, "sacre.tsv")

        with open(sacre_filename, "a", encoding='utf8') as sacre_file:
            sacre_file.write("NEW EPOCH:\n")

        eval_dataset_A_iter = enumerate(eval_dataset_A_mono.dataloader)
        eval_dataset_B_iter = enumerate(eval_dataset_B_mono.dataloader)
        for j in range(n):  # inner loop within one epoch
            _, eval_data_A = eval_dataset_A_iter.__next__()
            _, eval_data_B = eval_dataset_B_iter.__next__()
            model.set_input(eval_data_A, eval_data_B)  # unpack data from dataset and apply preprocessing
            model.evaluate(sentences_file=sentences_filename,sacre_file=sacre_filename)

        with open(sacre_filename, "a", encoding='utf8') as sacre_file:
            sacre_file.write("\n\n\n\n")

        logging.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
