import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
import copy
from utils import util, ssim_psnr
from IPython import embed
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
from thop import profile
from PIL import Image
import numpy as np

sys.path.append('../')
sys.path.append('./')
from interfaces import base
from utils.meters import AverageMeter
from utils.metrics import get_str_list, Accuracy
from utils.util import str_filt
from utils import utils_moran
import torch.optim as optim

class TextSR(base.TextBase):
    def train(self):
        cfg = self.config.TRAIN
        train_dataset, train_loader = self.get_train_data()
        val_dataset_list, val_loader_list = self.get_val_data()
        
        model_dict = self.generator_init()
        if self.config.MODEL.adversarial_epoch>=0:
            model, image_crit, netD = model_dict['model'], model_dict['crit'], model_dict['netD']  
        else: 
            model, image_crit, netD = model_dict['model'], model_dict['crit'], None

        rec_info = None
        if self.rec_metirc == 'crnn':
            rec_model = self.CRNN_init()
        elif self.rec_metirc == 'moran':
            rec_model = self.MORAN_init()
        elif self.rec_metirc == 'aster':
            rec_model, rec_info = self.Aster_init()
        else:
            rec_crnn = self.CRNN_init()
            rec_moran = self.MORAN_init()
            rec_aster, rec_info = self.Aster_init()
            rec_models = [rec_crnn, rec_moran, rec_aster]
            rec_infos = [None, None, rec_info]
            orders = ['crnn', 'moran', 'aster']

        optimizer_G = self.optimizer_init(model)
        optimizer_D = None
        if netD:
            optimizer_D = self.optimizer_init(netD)
            scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, 0.9, -1)

        best_history_acc = dict(
            zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                [0] * len(val_loader_list)))
        best_model_acc = copy.deepcopy(best_history_acc)
        best_model_psnr = copy.deepcopy(best_history_acc)
        best_model_ssim = copy.deepcopy(best_history_acc)
        best_acc = 0
        converge_list = []

        if self.rec_metirc == 'all':
            ttt = dict(
                zip([val_loader_dir.split('/')[-1] for val_loader_dir in self.config.TRAIN.VAL.val_data_dir],
                    [0] * len(val_loader_list)))
            crnn_acc = copy.deepcopy(ttt)
            aster_acc = copy.deepcopy(ttt)
            moran_acc = copy.deepcopy(ttt)
            best_history_acc = {'aster':aster_acc, 'crnn':crnn_acc, 'moran':moran_acc}
            best_model_acc = copy.deepcopy(best_history_acc)
            best_model_psnr = copy.deepcopy(best_history_acc)
            best_model_ssim = copy.deepcopy(best_history_acc)
            best_acc = {'aster':0, 'crnn':0, 'moran':0}
            converge_list = {'aster':[], 'crnn':[], 'moran':[]}

        for epoch in range(cfg.epochs):
            
            # train_bar = tqdm(train_loader)
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        
            model.train()
            netD.train() if netD else None

            for j, data in (enumerate(train_loader)):

                model.train()
                for p in model.parameters():
                    p.requires_grad = True
                iters = len(train_loader) * epoch + j + 1

                batch_size = data[0].size(0)
                running_results['batch_sizes'] += batch_size

                images_hr, images_lr, label_strs = data
                if self.args.syn:
                    images_lr = nn.functional.interpolate(images_hr, (self.config.TRAIN.height // self.scale_factor,
                                                                      self.config.TRAIN.width // self.scale_factor),
                                                          mode='bicubic')
                    images_lr = images_lr.to(self.device)
                else:
                    images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)

                

                image_sr = model(images_lr)
                total_loss = image_crit(image_sr, images_hr).mean() * 100

                optimizer_G.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer_G.step()

                if iters % cfg.displayInterval == 0:
                    print('[{}]\t'
                          'Epoch: [{}][{}/{}]\t'
                          'vis_dir={:s}\t'
                          'total_loss={:.3f} \t'
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  epoch, j + 1, len(train_loader),
                                  self.vis_root,
                                  float(total_loss.data)))



                # # adv version
                # ############################
                # # (1) Update D network: maximize D(x)-1-D(G(z))
                # ###########################
                
                # fake_img = model(images_lr)

                # if netD and epoch>=self.config.MODEL.adversarial_epoch and iters%10==0:
                #     netD.zero_grad()
                #     real_out = netD(images_hr).mean()
                #     fake_out = netD(fake_img).mean()
                #     d_loss = (1 - real_out + fake_out) * 0.001
                #     # optimizer_D.zero_grad()
                #     d_loss.backward(retain_graph=True)
                #     optimizer_D.step()
                #     if self.config.MODEL.optimizer_D_step>0 and iters % self.config.MODEL.optimizer_D_step == 0:
                #         scheduler_D.step()

                #     adversarial_loss = torch.mean(1 - fake_out)
                # else:
                #     d_loss = torch.tensor(0).cuda()
                #     real_out = torch.tensor(0).cuda()
                #     adversarial_loss = 0

                # ############################
                # # (2) Update G network: minimize 1-D(G(z))
                # ###########################
                # model.zero_grad()
                
                # total_loss = (image_crit(fake_img, images_hr).mean() + 0.01 * adversarial_loss)*100
                # # optimizer_G.zero_grad()
                # total_loss.backward()

                # # fake_img = model(images_lr)
                # fake_out = netD(fake_img).mean() if netD else torch.tensor(0).cuda()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                # optimizer_G.step()

                # loss for current batch before optimization 
                # running_results['g_loss'] += total_loss.item() * batch_size
                # running_results['d_loss'] += d_loss.item() * batch_size
                # running_results['d_score'] += real_out.item() * batch_size
                # running_results['g_score'] += fake_out.item() * batch_size
        
                # train_bar.set_description(desc='[%d/%d] Loss_D: %.5f Loss_G: %.3f D(x): %.3f D(G(z)): %.3f lr_D: %.6f' % (
                #     epoch, cfg.epochs, running_results['d_loss'] / running_results['batch_sizes'],
                #     running_results['g_loss'] / running_results['batch_sizes'],
                #     running_results['d_score'] / running_results['batch_sizes'],
                #     running_results['g_score'] / running_results['batch_sizes'],
                #     (optimizer_D.state_dict()['param_groups'][0]['lr']) if optimizer_D else -1))
                

                

            if (epoch + 1) >= cfg.VAL.valInterval_start and (epoch + 1) % cfg.VAL.valInterval_epochs == 0 and self.rec_metirc != 'all':
                print('======================================================')
                current_acc_dict = {}
                for k, val_loader in enumerate(val_loader_list):
                    data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                    print('[{}]\t'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    print('evaling %s' % data_name)
                    metrics_dict = self.eval_func(model, val_loader, image_crit, iters, rec_model, rec_info)
                    converge_list.append({'iterator': iters,
                                            'acc': metrics_dict['accuracy'],
                                            'psnr': metrics_dict['psnr_avg'],
                                            'ssim': metrics_dict['ssim_avg']})
                    acc = metrics_dict['accuracy']
                    current_acc_dict[data_name] = float(acc)
                    if acc > best_history_acc[data_name]:
                        best_history_acc[data_name] = float(acc)
                        best_history_acc['epoch'] = epoch
                        print('best_%s = %.2f%%*' % (data_name, best_history_acc[data_name] * 100))
                    else:
                        print('best_%s = %.2f%%' % (data_name, best_history_acc[data_name] * 100))
                    
                    self.tf_writer.add_scalar(data_name + '/acc', metrics_dict['accuracy'], iters)
                    # self.tf_writer.add_scalar(data_name + '/psnr', metrics_dict['psnr_avg'], iters)
                    # self.tf_writer.add_scalar(data_name + '/ssim', metrics_dict['ssim_avg'], iters)
            
                if self.calc_avg(current_acc_dict) > best_acc:
                    best_acc = self.calc_avg(current_acc_dict)
                    best_model_acc = current_acc_dict
                    best_model_acc['epoch'] = epoch
                    best_model_psnr[data_name] = metrics_dict['psnr_avg']
                    best_model_ssim[data_name] = metrics_dict['ssim_avg']
                    best_model_info = {'accuracy': best_model_acc, 'psnr': best_model_psnr, 'ssim': best_model_ssim}
                    print('saving best model, which avg is : %.2f%%' % (best_acc * 100))
                    self.save_checkpoint(model, epoch, iters, best_history_acc, best_model_info, True, converge_list)
                    
            if (epoch + 1) >= cfg.VAL.valInterval_start and (epoch + 1) % cfg.VAL.valInterval_epochs == 0 and self.rec_metirc == 'all':
                
                for rec_model, rec_info, order in zip(rec_models, rec_infos, orders):
                    print('======================================================')
                    print('[{}]\t'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                    current_acc_dict = {}
                    
                    for k, val_loader in enumerate(val_loader_list):
                        data_name = self.config.TRAIN.VAL.val_data_dir[k].split('/')[-1]
                        
                        print('evaling %s' % data_name)
                        metrics_dict = self.eval_func(model, val_loader, image_crit, iters, rec_model, rec_info, order)

                        converge_list[order].append({'iterator': iters,
                                                'acc': metrics_dict['accuracy'],
                                                'psnr': metrics_dict['psnr_avg'],
                                                'ssim': metrics_dict['ssim_avg']})
                        acc = metrics_dict['accuracy']
                        current_acc_dict[data_name] = float(acc)
                        if acc > best_history_acc[order][data_name]:
                            best_history_acc[order][data_name] = float(acc)
                            # best_history_acc[order]['epoch'] = epoch
                            print('best_%s = %.2f%%*' % (data_name, best_history_acc[order][data_name] * 100))
                        else:
                            print('best_%s = %.2f%%' % (data_name, best_history_acc[order][data_name] * 100))
                        
                        self.tf_writer.add_scalar(data_name + '/{}_acc'.format(order), metrics_dict['accuracy'], epoch)
                        self.tf_writer.add_scalar(data_name + '/psnr', metrics_dict['psnr_avg'], epoch)
                        self.tf_writer.add_scalar(data_name + '/ssim', metrics_dict['ssim_avg'], epoch)

                    self.tf_writer.add_scalar(order + '/average_acc', self.calc_avg(current_acc_dict), epoch)
                        
                    if self.calc_avg(current_acc_dict) > best_acc[order]:
                        best_acc[order] = self.calc_avg(current_acc_dict)
                        best_model_acc[order] = current_acc_dict
                        # best_model_acc[order]['epoch'] = epoch
                        best_model_psnr[order][data_name] = metrics_dict['psnr_avg']
                        best_model_ssim[order][data_name] = metrics_dict['ssim_avg']
                        best_model_info = {'accuracy': best_model_acc[order], 'psnr': best_model_psnr[order], 'ssim': best_model_ssim[order]}
                        print('saving best model, which avg is : %.2f%%' % (best_acc[order] * 100))
                        self.save_checkpoint(model, epoch, iters, best_history_acc[order], best_model_info, True, converge_list[order], order)

            self.tf_writer.add_scalar('g_loss', total_loss.item(), epoch)
            # self.tf_writer.add_scalar('d_loss', d_loss.item(), epoch)
            self.tf_writer.flush()
        
        print('saving best avg is :')
        print(best_acc)
        self.tf_writer.close()

 
    def calc_avg(self, acc_dict):
        hard_num = 1343
        med_num = 1411
        easy_num = 1619

        return (hard_num*acc_dict['hard'] + med_num*acc_dict['medium'] + easy_num*acc_dict['easy']) / (hard_num+med_num+easy_num)
    
    def eval_func(self, model, val_loader, image_crit, index, rec_model, rec_model_info=None, order=None):
        for p in rec_model.parameters():
            p.requires_grad = False

        # debug
        # model = torch.jit.load('/home/ubuntu/fsy_scenetext/PCAN/src/pcagan_4.pt').cuda()

        model.eval()
        rec_model.eval()
        n_correct = 0
        sum_images = 0
        
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        for i, data in (enumerate(val_loader)):
            images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]

            with torch.no_grad():
                images_sr = model(images_lr.to(self.device))
            
            if self.rec_metirc == 'moran' or (self.rec_metirc=='all' and order=='moran'):
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = rec_model(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
            elif self.rec_metirc == 'aster' or (self.rec_metirc=='all' and order=='aster'):
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = rec_model(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=rec_model_info)
            elif self.rec_metirc == 'crnn' or (self.rec_metirc=='all' and order=='crnn'):
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = rec_model(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
            # import ipdb;ipdb.set_trace()
            # TODO:if needed
            metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr.cuda()))
            metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr.cuda()))

            for pred, target in zip(pred_str_sr, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct += 1
            
            sum_images += val_batch_size
            torch.cuda.empty_cache()

        # TODO:if needed
        psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        
        if self.config.TRAIN.VAL.n_vis > 0:
            print('save display images')
            self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)
        
        accuracy = round(n_correct / sum_images, 4)
        if order is not None:
            print(order+'_accuray: %.2f%%' % (accuracy * 100))
        else:
            print(self.rec_metirc+'_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = psnr_avg
        metric_dict['ssim_avg'] = ssim_avg
        
        model.train()
        return metric_dict
    
    def eval_crnn(self, model, val_loader, image_crit, index, crnn):
        for p in crnn.parameters():
            p.requires_grad = False
        model.eval()
        crnn.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        for i, data in (enumerate(val_loader)):
            if self.config.MODEL is not None and self.config.MODEL.get('language_model', False):
                images_hr, images_lr, label_strs, embed_vec = data
            else:
                images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]

            with torch.no_grad():
                images_sr = model(data)

            crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
            crnn_output = crnn(crnn_input)
            _, preds = crnn_output.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
            pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

            for pred, target in zip(pred_str_sr, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct += 1
            
            sum_images += val_batch_size
            torch.cuda.empty_cache()
        
        if self.config.TRAIN.VAL.n_vis > 0:
            print('save display images')
            self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)
        
        accuracy = round(n_correct / sum_images, 4)
        print('aster_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        metric_dict['psnr_avg'] = 0.0
        metric_dict['ssim_avg'] = 0.0
        
        model.train()
        return metric_dict
        
    def eval(self, model, val_loader, image_crit, index, aster, aster_info=None):
        for p in aster.parameters():
            p.requires_grad = False
        model.eval()
        aster.eval()
        n_correct = 0
        sum_images = 0
        metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
        for i, data in (enumerate(val_loader)):
            if self.config.MODEL is not None and self.config.MODEL.get('language_model', False):
                images_hr, images_lr, label_strs, embed_vec = data
            else:
                images_hr, images_lr, label_strs = data
            val_batch_size = images_lr.shape[0]
            # images_lr = images_lr.to(self.device)
            # images_hr = images_hr.to(self.device)
            
            # images_sr = model(images_lr)
            with torch.no_grad():
                images_sr = model(data)
            
            # TODO:if needed
            # metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
            # metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))
            
            aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
            # aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
            # aster_dict_hr = self.parse_aster_data(images_hr[:, :3, :, :])
            
            aster_output_sr = aster(aster_dict_sr)
            # aster_output_lr = aster(aster_dict_lr)
            # aster_output_hr = aster(aster_dict_hr)
            
            pred_rec_sr = aster_output_sr['output']['pred_rec']
            # pred_rec_lr = aster_output_lr['output']['pred_rec']
            # pred_rec_hr = aster_output_hr['output']['pred_rec']
            
            pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)
            # pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            # pred_str_hr, _ = get_str_list(pred_rec_hr, aster_dict_hr['rec_targets'], dataset=aster_info)
            
            for pred, target in zip(pred_str_sr, label_strs):
                if pred == str_filt(target, 'lower'):
                    n_correct += 1

            # TODO:if needed
            # loss_im = image_crit(images_sr, images_hr).mean()
            # loss_rec = aster_output_sr['losses']['loss_rec'].mean()
            
            sum_images += val_batch_size
            torch.cuda.empty_cache()
            
        # TODO:if needed
        # psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
        # ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
        # print('[{}]\t'
        #       'loss_rec {:.3f}| loss_im {:.3f}\t'
        #       'PSNR {:.2f} | SSIM {:.4f}\t'
        #       .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        #               float(loss_rec.data), 0,
        #               float(psnr_avg), float(ssim_avg), ))
        
        if self.config.TRAIN.VAL.n_vis > 0:
            print('save display images')
            self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)
        
        accuracy = round(n_correct / sum_images, 4)
        # psnr_avg = round(psnr_avg.item(), 6)
        # ssim_avg = round(ssim_avg.item(), 6)
        print('aster_accuray: %.2f%%' % (accuracy * 100))
        metric_dict['accuracy'] = accuracy
        # metric_dict['psnr_avg'] = psnr_avg
        # metric_dict['ssim_avg'] = ssim_avg
        metric_dict['psnr_avg'] = 0.0
        metric_dict['ssim_avg'] = 0.0
        
        model.train()
        return metric_dict

    def test(self):
        
        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        
        # import ipdb;ipdb.set_trace()
        # sr_model_path = '/home/ubuntu/fsy_scenetext/FoolText-master/defence_attack/pcan/src/pcagan_1.pt'
        # model = torch.jit.load(sr_model_path, map_location=torch.device('cuda:0'))

        current_acc_dict = {}
        for idx, cur_test_data_dir in enumerate(self.test_data_dir):
            test_data, test_loader = self.get_test_data(cur_test_data_dir)
            data_name = cur_test_data_dir.split('/')[-1]
            print('evaling %s' % data_name)
            
            # if self.rec_metirc == 'moran':
            #     moran = self.MORAN_init()
            #     moran.eval()
            # elif self.rec_metirc == 'aster':
            #     aster, aster_info = self.Aster_init()
            #     aster.eval()
            # elif self.rec_metirc == 'crnn':
            #     crnn = self.CRNN_init()
            #     crnn.eval()
            
            aster, aster_info = self.Aster_init()
            aster.eval()
            
            
            
            # print(sum(p.numel() for p in moran.parameters()))
            if self.config.MODEL.arch != 'bicubic':
                for p in model.parameters():
                    p.requires_grad = False
                model.eval()

            n_correct = 0
            sum_images = 0
            metric_dict = {'psnr': [], 'ssim': [], 'accuracy': 0.0, 'psnr_avg': 0.0, 'ssim_avg': 0.0}
            
            time_begin = time.time()
            sr_time = 0
            
            for i, data in (enumerate(test_loader)):
                images_hr, images_lr, label_strs = data
                val_batch_size = images_lr.shape[0]
                images_lr = images_lr.to(self.device)
                images_hr = images_hr.to(self.device)
                sr_beigin = time.time()

                # import ipdb;ipdb.set_trace()
                # demo = torch.rand((1, 4, 16, 64))
                # torchjit_model = torch.jit.trace(model, demo.cuda())
                # torch.jit.save(torchjit_model, 'pcagan_9.pt')
                
                images_sr = model(images_lr)
                if self.config.MODEL.arch == 'bicubic':
                    images_sr = images_sr.to(self.device)

                # images_sr = images_lr
                sr_end = time.time()
                sr_time += sr_end - sr_beigin
                metric_dict['psnr'].append(self.cal_psnr(images_sr, images_hr))
                metric_dict['ssim'].append(self.cal_ssim(images_sr, images_hr))

                
                
                # if self.args.rec == 'moran':
                #     moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                #     moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                #                         debug=True)
                #     preds, preds_reverse = moran_output[0]
                #     _, preds = preds.max(1)
                #     sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                #     pred_str_sr = [pred.split('$')[0] for pred in sim_preds]
                # elif self.args.rec == 'aster':
                #     aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                #     aster_output_sr = aster(aster_dict_sr)
                #     pred_rec_sr = aster_output_sr['output']['pred_rec']
                #     pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                #     aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                #     aster_output_lr = aster(aster_dict_lr)
                #     pred_rec_lr = aster_output_lr['output']['pred_rec']
                #     pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                # elif self.args.rec == 'crnn':
                #     crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                #     crnn_output = crnn(crnn_input)
                #     _, preds = crnn_output.max(2)
                #     preds = preds.transpose(1, 0).contiguous().view(-1)
                #     preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                #     pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)
                
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
                
                for pred, target in zip(pred_str_sr, label_strs):
                    if str_filt(pred, 'lower') == str_filt(target, 'lower'):
                        n_correct += 1
                sum_images += val_batch_size
                torch.cuda.empty_cache()
                print('Evaluation: [{}][{}/{}]\t'
                    .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            i + 1, len(test_loader), ))
                
                # import ipdb;ipdb.set_trace()
                # self.test_display(images_lr[:, :3, :, :], images_sr[:, :3, :, :], images_hr[:, :3, :, :], pred_str_lr, pred_str_sr, label_strs, str_filt, data_name, i)
            time_end = time.time()
            psnr_avg = sum(metric_dict['psnr']) / len(metric_dict['psnr'])
            ssim_avg = sum(metric_dict['ssim']) / len(metric_dict['ssim'])
            acc = round(n_correct / sum_images, 4)
            fps = sum_images/(time_end - time_begin)
            psnr_avg = round(psnr_avg.item(), 6)
            ssim_avg = round(ssim_avg.item(), 6)
            current_acc_dict[data_name] = float(acc)
            # result = {'accuracy': current_acc_dict, 'fps': fps}
        # import ipdb;ipdb.set_trace()
        result = {'accuracy': current_acc_dict, 'avg': self.calc_avg(current_acc_dict), 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg, 'fps': fps}
        print(result)

    def demo(self):
        mask_ = self.args.mask

        def transform_(path):
            img = Image.open(path)
            img = img.resize((256, 32), Image.BICUBIC)
            img_tensor = transforms.ToTensor()(img)
            if mask_:
                mask = img.convert('L')
                thres = np.array(mask).mean()
                mask = mask.point(lambda x: 0 if x > thres else 255)
                mask = transforms.ToTensor()(mask)
                img_tensor = torch.cat((img_tensor, mask), 0)
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor

        model_dict = self.generator_init()
        model, image_crit = model_dict['model'], model_dict['crit']
        if self.args.rec == 'moran':
            moran = self.MORAN_init()
            moran.eval()
        elif self.args.rec == 'aster':
            aster, aster_info = self.Aster_init()
            aster.eval()
        elif self.args.rec == 'crnn':
            crnn = self.CRNN_init()
            crnn.eval()
        if self.args.arch != 'bicubic':
            for p in model.parameters():
                p.requires_grad = False
            model.eval()
        n_correct = 0
        sum_images = 0
        time_begin = time.time()
        sr_time = 0
        for im_name in tqdm(os.listdir(self.args.demo_dir)):
            images_lr = transform_(os.path.join(self.args.demo_dir, im_name))
            images_lr = images_lr.to(self.device)
            sr_beigin = time.time()
            images_sr = model(images_lr)

            sr_end = time.time()
            sr_time += sr_end - sr_beigin
            if self.args.rec == 'moran':
                moran_input = self.parse_moran_data(images_sr[:, :3, :, :])
                moran_output = moran(moran_input[0], moran_input[1], moran_input[2], moran_input[3], test=True,
                                     debug=True)
                preds, preds_reverse = moran_output[0]
                _, preds = preds.max(1)
                sim_preds = self.converter_moran.decode(preds.data, moran_input[1].data)
                pred_str_sr = [pred.split('$')[0] for pred in sim_preds]

                moran_input_lr = self.parse_moran_data(images_lr[:, :3, :, :])
                moran_output_lr = moran(moran_input_lr[0], moran_input_lr[1], moran_input_lr[2], moran_input_lr[3], test=True,
                                     debug=True)
                preds_lr, preds_reverse_lr = moran_output_lr[0]
                _, preds_lr = preds_lr.max(1)
                sim_preds_lr = self.converter_moran.decode(preds_lr.data, moran_input_lr[1].data)
                pred_str_lr = [pred.split('$')[0] for pred in sim_preds_lr]
            elif self.args.rec == 'aster':
                aster_dict_sr = self.parse_aster_data(images_sr[:, :3, :, :])
                aster_output_sr = aster(aster_dict_sr)
                pred_rec_sr = aster_output_sr['output']['pred_rec']
                pred_str_sr, _ = get_str_list(pred_rec_sr, aster_dict_sr['rec_targets'], dataset=aster_info)

                aster_dict_lr = self.parse_aster_data(images_lr[:, :3, :, :])
                aster_output_lr = aster(aster_dict_lr)
                pred_rec_lr = aster_output_lr['output']['pred_rec']
                pred_str_lr, _ = get_str_list(pred_rec_lr, aster_dict_lr['rec_targets'], dataset=aster_info)
            elif self.args.rec == 'crnn':
                crnn_input = self.parse_crnn_data(images_sr[:, :3, :, :])
                crnn_output = crnn(crnn_input)
                _, preds = crnn_output.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output.size(0)] * val_batch_size)
                pred_str_sr = self.converter_crnn.decode(preds.data, preds_size.data, raw=False)

                crnn_input_lr = self.parse_crnn_data(images_lr[:, :3, :, :])
                crnn_output_lr = crnn(crnn_input_lr)
                _, preds_lr = crnn_output_lr.max(2)
                preds_lr = preds_lr.transpose(1, 0).contiguous().view(-1)
                preds_size = torch.IntTensor([crnn_output_lr.size(0)] * val_batch_size)
                pred_str_lr = self.converter_crnn.decode(preds_lr.data, preds_size.data, raw=False)
            print(pred_str_lr, '===>', pred_str_sr)
            torch.cuda.empty_cache()
        sum_images = len(os.listdir(self.args.demo_dir))
        time_end = time.time()
        fps = sum_images / (time_end - time_begin)
        print('fps=', fps)



if __name__ == '__main__':
    embed()
