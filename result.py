import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import json

def parse_single_whitebox(json_file):
    with open(json_file, 'r') as fp:
        j = json.load(fp)
        return [j[0]['AE clean'],
                j[2]['FGSM']['whiteadv_rec'],
                j[3]['PGD']['whiteadv_rec'],
                j[1]['CW']['whiteadv_rec']]
    

def parse_result(advae_json, advtrain_json, hgd_json):
    with open(advae_json, 'r') as fp:
        advae = json.load(fp)

    res = {}
    res['nodef'] = [advae[0]['CNN clean'],
                    advae[3]['PGD']['obliadv'],
                    advae[2]['FGSM']['obliadv'],
                    advae[1]['CW']['obliadv']]
    
    res['advae obli'] = [-1,
                         advae[3]['PGD']['obliadv_rec'],
                         advae[2]['FGSM']['obliadv_rec'],
                         advae[1]['CW']['obliadv_rec']]
    
    res['advae whitebox'] = parse_single_whitebox(advae_json)
    res['hgd whitebox'] = parse_single_whitebox(hgd_json)
    res['advtrain whitebox'] = parse_single_whitebox(advtrain_json)
    return res


def parse_mnist_table(advae_json, advtrain_json, hgd_json, bbox_json, defgan_json):
    """Parse result for MNIST table.

    Rows (attacks)
    - no attack
    - FGSM
    - PGD
    - CW L2

    Columns (defences):
    - no defence
    - AdvAE oblivious
    - AdvAE blackbox
    - AdvAE whitebox
    - HGD whitebox
    - DefenseGAN whitebox
    - adv training

    """
    with open(advae_json, 'r') as fp:
        advae = json.load(fp)
    with open(bbox_json, 'r') as fp:
        bbox = json.load(fp)
    with open(defgan_json, 'r') as fp:
        defgan = json.load(fp)
    res = {}
    res['nodef'] = [advae[0]['CNN clean'],
                    advae[2]['FGSM']['obliadv'],
                    advae[3]['PGD']['obliadv'],
                    advae[1]['CW']['obliadv']]
    
    res['advae obli'] = [0,
                         # FIXME adjust json format to remove this
                         # indices
                         advae[2]['FGSM']['obliadv_rec'],
                         advae[3]['PGD']['obliadv_rec'],
                         advae[1]['CW']['obliadv_rec']]
    res['advae bbox'] = [0,
                         bbox['FGSM'],
                         bbox['PGD'],
                         bbox['CW']]
    
    res['advae whitebox'] = parse_single_whitebox(advae_json)
    res['hgd whitebox'] = parse_single_whitebox(hgd_json)
    res['defgan whitebox'] = [defgan['CNN clean'],
                              defgan['FGSM'],
                              defgan['PGD'],
                              defgan['CW']]
    res['advtrain whitebox'] = parse_single_whitebox(advtrain_json)
    # print out the table in latex format

    for i, name in enumerate(['No attack', 'FGSM', 'PGD', 'CW $\ell_2$']):
        print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\\\\'
              .format(name, *[res[key][i]*100 for key in res]))
    return


def parse_mnist_transfer_table(trans_jsons, ensemble_jsons):
    """Table for transfer and ensemble models.

    We only evaluate whitebox here.

    Rows (attacks)
    - no attack
    - FGSM
    - PGD
    - CW L2

    Columns (defences setting):
    - X/A
    - X/B
    - X/C
    - X,A,B/A
    - X,A,B/B
    - X,A,B/C
    - X,A,B/D
    """
    trans = []
    ensemble = []
    for tj in trans_jsons:
        trans.append(parse_single_whitebox(tj))
    # for ej in ensemble_jsons:
    #     ensemble.append(parse_single_whitebox(ej))
    for i, name in enumerate(['No attack', 'FGSM', 'PGD', 'CW $\ell_2$']):
        print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & '
              .format(name, *[r[i]*100 for r in trans]), end=' ')
        # print('{:.1f} & {:.1f} & {:.1f} & {:.1f}\\\\'
        #       .format(*[r[i]*100 for r in ensemble]))
        print('')

def parse_lambda_table():
    fmt = 'images/test-result-MNIST-mnistcnn-ae-C0_A2_{}.json'
    fmt2 = 'images/test-result-MNIST-mnistcnn-ae-C0_A2_{}-TO-DefenseGAN_d.json'
    # fmt = 'images/test-result-CIFAR10-resnet29-dunet-C0_A2_{}.json'
    lambdas = [0, 0.2, 0.5, 1, 1.5, 5]
    res = {}
    res['FGSM'] = []
    res['PGD'] = []
    res['PGD X/D'] = []
    res['no'] = []
    res['CW'] = []
    for l in lambdas:
        with open(fmt.format(l), 'r') as fp:
            j = json.load(fp)
            res['no'].append(j[0]['AE clean'])
            res['PGD'].append(j[3]['PGD']['whiteadv_rec'])
            res['FGSM'].append(j[2]['FGSM']['whiteadv_rec'])
            res['CW'].append(j[1]['CW']['whiteadv_rec'])
        with open(fmt2.format(l), 'r') as fp:
            j = json.load(fp)
            res['PGD X/D'].append(j[3]['PGD']['whiteadv_rec'])
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('No attack', *[r*100 for r in res['no']]), end=' ')
    print('')
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('PGD', *[r*100 for r in res['PGD']]), end=' ')
    print('')
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('FGSM', *[r*100 for r in res['FGSM']]), end=' ')
    print('')
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('CW', *[r*100 for r in res['CW']]), end=' ')
    print('')
    print('{} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}'
          .format('PGD X/D', *[r*100 for r in res['PGD X/D']]), end=' ')
    # return res

def __test():
    # extracting experiment results

    parse_lambda_table()
    parse_single_whitebox('images/test-result-MNIST-mnistcnn-ae-C0_A2_1.json')

    # MNIST
    parse_mnist_table(
        'images/test-result-MNIST-mnistcnn-ae-A2.json',
        'images/test-result-MNIST-mnistcnn-identityAE-A2.json',
        'images/test-result-MNIST-mnistcnn-ae-B2.json',
        'images/test-result-MNIST-mnistcnn-ae-A2-BBOX-mnistcnn.json',
        'images/defgan.json')

    parse_mnist_transfer_table(
        ['images/test-result-MNIST-mnistcnn-ae-C0_A2_0-TO-DefenseGAN_a.json',
         'images/test-result-MNIST-mnistcnn-ae-C0_A2_0-TO-DefenseGAN_b.json',
         'images/test-result-MNIST-mnistcnn-ae-C0_A2_0-TO-DefenseGAN_c.json',
         'images/test-result-MNIST-mnistcnn-ae-C0_A2_0-TO-DefenseGAN_d.json'],
        ['images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-ae-A2-ENSEMBLE-DefenseGAN_a.json',
         'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-ae-A2-ENSEMBLE-DefenseGAN_b.json',
         'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-ae-A2-ENSEMBLE-DefenseGAN_c.json',
         'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-ae-A2-ENSEMBLE-DefenseGAN_d.json'
        ])

    # fashion
    res = parse_result('images/test-result-Fashion-mnistcnn-ae-A2.json',
                       'images/test-result-Fashion-mnistcnn-identityAE-A2.json',
                       'images/test-result-Fashion-mnistcnn-ae-B2.json')
    # cifar
    res = parse_result('images/test-result-CIFAR10-resnet29-dunet-C0_A2.json',
                       'images/test-result-CIFAR10-resnet29-identityAE-A2.json',
                       'images/test-result-CIFAR10-resnet29-dunet-C0_B2.json')
    # transfer models
    res = {}
    for defgan_var in ['a', 'b', 'c', 'd', 'e', 'f']:
        # fname = 'images/test-result-{}-TransTo-DefenseGAN_{}-ae-A2.json'.format('mnist', defgan_var)
        # fname = 'images/test-result-{}-TO-DefenseGAN_{}.json'.format('MNIST-mnistcnn-ae-A2', defgan_var)
        # fname = 'images/test-result-{}-TO-DefenseGAN_{}.json'.format('Fashion-mnistcnn-ae-A2', defgan_var)
        fname = 'images/test-result-{}-TO-DefenseGAN_{}.json'.format('CIFAR10-resnet29-dunet-C0_A2', defgan_var)
        res[defgan_var] = parse_single_whitebox(fname)

    # resnet to WRN
    fname = 'images/test-result-{}-TO-{}.json'.format('CIFAR10-resnet29-dunet-C0_A2', 'WRN')
    parse_single_whitebox(fname)
        
    for i in range(4):
        for k in res:
            print('{:.3f}'.format(res[k][i]), end=',')
        print()

def myplot():

    # TODO plot
    with open('images/epsilon-advae-100.json') as fp:
        advae = json.load(fp)
    with open('images/epsilon-hgd-100-fix.json') as fp:
        hgd = json.load(fp)
    with open('images/epsilon-itadv-100.json') as fp:
        itadv = json.load(fp)

    fig = plt.figure(
        # figsize=(fig_width, fig_height),
        dpi=300)
    
    # fig.canvas.set_window_title('My Grid Visualization')
    plt.plot(hgd['eps'], hgd['PGD'], 'x-', color='green', markersize=4, label='PGD / HGD')
    plt.plot(hgd['eps'], hgd['Hop'], 'x', dashes=[2, 3],  color='green', markersize=4, label='HSJA / HGD')
    plt.plot(itadv['eps'], itadv['PGD'], '^-', color='brown', markersize=4, label='PGD / ItAdv')
    plt.plot(itadv['eps'], itadv['Hop'], '^', dashes=[2,3], color='brown', markersize=4, label='HSJA / ItAdv')
    plt.plot(advae['eps'], advae['PGD'], 'o-', color='blue', markersize=4,
             label='PGD / ' + r"$\bf{" + 'AdvAE' + "}$")
    plt.plot(advae['eps'], advae['Hop'], 'o', dashes=[2,3], color='blue', markersize=4,
             label='HSJA / ' + r"$\bf{" + 'AdvAE' + "}$")
    # HopSkipJump
    plt.xlabel('Distortion')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    # plt.title('Accuracy against PGD and HSJA on different distortions')
    
    # if titles is None:
    #     titles = [''] * len(images)

    # for image, ax, title in zip(images, axes.reshape(-1), titles):
    #     # ax.set_axis_off()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #     # TITLE has to appear on top. I want it to be on bottom, so using xlabel
    #     ax.set_title(title, loc='left')
    #     # ax.set_xlabel(title)
    #     # ax.imshow(convert_image_255(image), cmap='gray')
    #     # ax.imshow(image)
    #     ax.imshow(image, cmap='gray')
    
    # plt.subplots_adjust(hspace=0.5)
    plt.savefig('test.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
def __test_plot():
    myplot()

def plot_onto_lambda():
    "With ep=0.38"
    lams = [0, 0.2, 0.5, 1, 1.5, 2, 5]
    res = []
    for lam in lams:
        fname = 'images/test-result-MNIST-mnistcnn-cnn3AE-C0_A2_{}.json'.format(lam)
        with open(fname) as fp:
            j = json.load(fp)
            data = j['epsilon_exp_data']
            assert round(data[2][0], ndigits=1) == 0.1
            assert round(data[7][0], ndigits=1) == 0.3
            assert round(data[9][0], ndigits=2) == 0.38
            assert round(data[12][0], ndigits=1) == 0.5
            # FGSM, PGD, Hop
            res.append([data[9][1], data[9][2], data[9][3]])
    fig = plt.figure(dpi=300)

    plt.plot(lams, [d[0] for d in res], 'x-', color='green', markersize=4, label='FGSM')
    plt.plot(lams, [d[1] for d in res], '^-', color='brown', markersize=4, label='PGD')
    plt.plot(lams, [d[2] for d in res], 'o-', color='blue', markersize=4, label='HSJA')

    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.savefig('images/onto-lambda-epsilon-0.38.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_lambda_onto_epsilon():
    """Plot lambda figure, for a selected epsilon setting."""
    lams = [0, 0.2, 0.5, 1, 1.5, 2, 5]
    lam_data = []
    for lam in lams:
        fname = 'images/test-result-MNIST-mnistcnn-cnn3AE-C0_A2_{}.json'.format(lam)
        res = {}
        with open(fname) as fp:
            j = json.load(fp)
            data = j['epsilon_exp_data']
            res['eps'] = [d[0] for d in data]
            res['FGSM'] = [d[1] for d in data]
            res['PGD'] = [d[2] for d in data]
            res['Hop'] = [d[3] for d in data]
        lam_data.append(res)

    fig = plt.figure(dpi=300)

    colors = [(0.5, x, y) for x,y in zip(np.arange(0, 1, 1 / len(lams)),
                                         np.arange(0, 1, 1 / len(lams)))]
    # colors2 = [(0.2, x, y) for x,y in zip(range(0, 1, len(lams)),
    #                                       range(0, 1, len(lams)))]
    for lam, data, marker, color in zip(lams, lam_data,
                                         # ['x-', '^-', 'o-', 's-']*2,
                                         ['o-']*len(lams),
                                         colors):
        # plt.plot(data['eps'], data['FGSM'], marker, color=color,
        #          markersize=4, label='FGSM {}'.format(lam))
        plt.plot(data['eps'], data['PGD'], marker, color=color,
                 markersize=4, label='PGD {}'.format(lam))
        # plt.plot(data['eps'], data['Hop'], marker, color=color2,
        #          markersize=4, label='Hop {}'.format(lam))
    plt.xlabel('Distortion')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.savefig('images/lamda-onto-epsilon.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def __test():
    # this is the main lambda plot
    plot_lambda_onto_epsilon()
    plot_onto_lambda()
