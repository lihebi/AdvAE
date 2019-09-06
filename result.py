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
    lams = [0.5, 0.8, 1, 1.2, 1.5, 2, 3, 4, 5]
    # lams = [0, 0.2, 0.5, 1, 1.5, 2, 5]
    res = []
    for lam in lams:
        fname = 'images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_{}.json'.format(lam)
        with open(fname) as fp:
            j = json.load(fp)
            # FGSM, PGD, Hop
            res.append(j['PGD'][7])
    fig = plt.figure(dpi=300)

    plt.plot(lams, res, 'x-', color='green', markersize=4, label='PGD')
    # plt.plot(lams, [d[1] for d in res], '^-', color='brown', markersize=4, label='PGD')
    # plt.plot(lams, [d[2] for d in res], 'o-', color='blue', markersize=4, label='HSJA')

    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.savefig('images/onto-lambda-epsilon.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_defense_onto_epsilon():
    "HGD, ItAdv, AdvAE"
    
    jfiles = ['images/test-result-MNIST-mnistcnn-cnn1AE-B2.json',
              'images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_1.json',
              'images/test-result-MNIST-mnistcnn-identityAE-ItAdv.json']
    lams = ['HGD',
            r"$\bf{"  + 'AdvAE' + "}$",
            'ItAdv']
    assert len(jfiles) == len(lams)
    lam_data = []
    for fname in jfiles:
        res = {}
        with open(fname) as fp:
            j = json.load(fp)
            res['eps'] = j['epsilon']
            res['FGSM'] = j['FGSM']
            res['PGD'] = j['PGD']
            res['Hop'] = j['Hop']
        lam_data.append(res)

    colors = [(0.5, x, y) for x,y in zip(np.arange(0, 1, 1 / len(lam_data)),
                                         np.arange(0, 1, 1 / len(lam_data)))]
    # I'm going to omit FGSM, and plot both PGD and Hop onto the same figure
    fig = plt.figure(dpi=300)
    
    for lam, data, marker, color in zip(lams, lam_data,
                                               ['x', '^', 'o'],
                                               ['brown', 'green', 'blue']):
        plt.plot(data['eps'], data['PGD'], marker + '-', color=color,
                 markersize=4, label='{} / {}'.format('PGD', lam))
        plt.plot(data['eps'], data['Hop'], marker, dashes=[2,3], color=color,
                 markersize=4, label='{} / {}'.format('HSJA', lam))
    plt.axvline(x=.3, ymin=0.05, ymax=0.95, color='red', alpha=0.5, linewidth=0.5)
    plt.axvline(x=.38, ymin=0.05, ymax=0.95, color='red', alpha=0.5, linewidth=0.5)
    plt.axvline(x=.46, ymin=0.05, ymax=0.95, color='red', alpha=0.5, linewidth=0.5)
    plt.xlabel('Distortion')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.savefig('images/defense-onto-epsilon.pdf',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_defense_onto_epsilon_cifar():
    "HGD, ItAdv, AdvAE"
    
    jfiles = ['images/test-result-CIFAR10-resnet29-dunet-B2.json',
              'images/test-result-CIFAR10-resnet29-dunet-C0_A2_2.json',
              'images/test-result-CIFAR10-resnet29-identityAE-ItAdv.json']
    lams = ['HGD',
            r"$\bf{"  + 'AdvAE' + "}$",
            'ItAdv']
    assert len(jfiles) == len(lams)
    lam_data = []
    for fname in jfiles:
        res = {}
        with open(fname) as fp:
            j = json.load(fp)
            res['eps'] = j['epsilon']
            res['FGSM'] = j['FGSM']
            res['PGD'] = j['PGD']
            res['Hop'] = j['Hop']
        lam_data.append(res)

    colors = [(0.5, x, y) for x,y in zip(np.arange(0, 1, 1 / len(lam_data)),
                                         np.arange(0, 1, 1 / len(lam_data)))]
    # I'm going to omit FGSM, and plot both PGD and Hop onto the same figure
    fig = plt.figure(dpi=300)
    
    for lam, data, marker, color in zip(lams, lam_data,
                                               ['x', '^', 'o'],
                                               ['brown', 'green', 'blue']):
        plt.plot(data['eps'], data['PGD'], marker + '-', color=color,
                 markersize=4, label='{} / {}'.format('PGD', lam))
        plt.plot(data['eps'], data['Hop'], marker, dashes=[2,3], color=color,
                 markersize=4, label='{} / {}'.format('HSJA', lam))
    plt.axvline(x=8/255, ymin=0.05, ymax=0.95, color='red', alpha=0.5, linewidth=0.5)
    plt.axvline(x=14/255, ymin=0.05, ymax=0.95, color='red', alpha=0.5, linewidth=0.5)
    plt.axvline(x=20/255, ymin=0.05, ymax=0.95, color='red', alpha=0.5, linewidth=0.5)
    plt.xlabel('Distortion')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.savefig('images/defense-onto-epsilon_cifar.pdf',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_lambda_onto_epsilon():
    """Plot lambda figure, for a selected epsilon setting."""
    # lams = [0, 0.2, 0.5, 1, 1.5, 2, 5]
    lams = [0, 0.2, 0.5, 0.8, 1, 1.2, 1.5, 2, 3, 4, 5]
    # lams = [0.5, 0.8, 1, 1.2, 1.5, 2, 3, 4, 5]
    lam_data = []
    for lam in lams:
        fname = 'images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_{}.json'.format(lam)
        res = {}
        with open(fname) as fp:
            j = json.load(fp)
            res['eps'] = j['epsilon']
            res['FGSM'] = j['FGSM']
            res['PGD'] = j['PGD']
            # res['Hop'] = j['Hop']
        lam_data.append(res)

    colors = [(0.5, x, y) for x,y in zip(np.arange(0, 1, 1 / len(lams)),
                                         np.arange(0, 1, 1 / len(lams)))]
    # colors2 = [(0.2, x, y) for x,y in zip(range(0, 1, len(lams)),
    #                                       range(0, 1, len(lams)))]

    for attack in ['FGSM', 'PGD']:
        fig = plt.figure(dpi=300)
        for lam, data, marker, color in zip(lams, lam_data,
                                            ['o-', '*-', 'x-', 's-', '^-'] * 3,
                                            colors):
            plt.plot(data['eps'], data[attack], marker, color=color,
                     markersize=4, label='{} $\lambda$={}'.format(attack, lam))
        plt.xlabel('Distortion')
        plt.ylabel('Accuracy')
        plt.legend(fontsize='small')
        plt.savefig('images/lambda-onto-epsilon-{}.pdf'.format(attack),
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def plot_aesize_onto_epsilon():
    # cnn_params = 1111946
    # ae_params = [50992, 222384, 2625, 3217, 4385]
    fig = plt.figure(dpi=300)
    colors = [(0.5, x, y) for x,y in zip(np.arange(0, 1, 1 / 9),
                                         np.arange(0, 1, 1 / 9))]
    for ae, name, marker, color in zip(['fcAE', 'deepfcAE', 'cnn1AE', 'cnn2AE', 'cnn3AE',
                                        'wide_16_32_32_16_AE',
                                        'wide_32_16_16_32_AE',
                                        'wide_32_32_32_32_AE',
                                        'wide_32_64_64_32_AE'
    ],
                                       ['1-layer 32 FC', '5-layer 128-64-32-64-128 FC',
                                        '2-layer 16-16 CNN',
                                        '4-layer 16-8-8-16 CNN',
                                        '6-layer 16-8-8-8-8-16 CNN',
                                        '4-layer 16-32-32-16',
                                        '4-layer 32_16_16_32',
                                        '4-layer 32_32_32_32',
                                        '4-layer 32_64_64_32'
                                       ],
                                       ['x-', '^-', 'o-', 's-']*3,
                                       colors):
        fname = 'images/test-result-MNIST-mnistcnn-{}-C0_A2_1.json'.format(ae)
        with open(fname) as fp:
            j = json.load(fp)
            eps = j['epsilon']
            pgd_data = j['PGD']
            param = j['AE params']
            plt.plot(eps, pgd_data, marker, color=color,
                     markersize=4, label='{}, #param={}'.format(name, param))
    plt.xlabel('Distortion')
    plt.ylabel('Accuracy')
    plt.legend(fontsize='small')
    plt.savefig('images/aesize-onto-epsilon.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def plot_train_process():
    fname = 'images/train-process.json'
    fig = plt.figure(dpi=300)
    keys1 = [
        # different defense
        'MNIST-mnistcnn-cnn1AE-ItAdv-AdvAE',
        'MNIST-mnistcnn-cnn3AE-B2-AdvAE',
        'MNIST-mnistcnn-cnn3AE-ItAdv-AdvAE',
        'MNIST-mnistcnn-identityAE-ItAdv-AdvAE',
    ]
    keys2 = [
        # differet AE
        'MNIST-mnistcnn-cnn1AE-C0_A2_1-AdvAE',
        'MNIST-mnistcnn-cnn2AE-C0_A2_1-AdvAE',
        'MNIST-mnistcnn-cnn3AE-C0_A2_1-AdvAE',
        'MNIST-mnistcnn-fcAE-C0_A2_1-AdvAE',
        'MNIST-mnistcnn-deepfcAE-C0_A2_1-AdvAE',
    ]
    keys3 = [
        # different lambdas
        # 'MNIST-mnistcnn-cnn3AE-C0_A2_0-AdvAE',
        # 'MNIST-mnistcnn-cnn3AE-C0_A2_0.2-AdvAE',
        'MNIST-mnistcnn-cnn3AE-C0_A2_0.5-AdvAE',
        'MNIST-mnistcnn-cnn3AE-C0_A2_1-AdvAE',
        'MNIST-mnistcnn-cnn3AE-C0_A2_1.5-AdvAE',
        'MNIST-mnistcnn-cnn3AE-C0_A2_2-AdvAE',
        'MNIST-mnistcnn-cnn3AE-C0_A2_5-AdvAE'
    ]

    keys = keys2

    markers = ['o-', '*-', 'x-', '^-'] * 3
    with open(fname) as fp:
        j = json.load(fp)
        colors = [(0.5, x, y) for x,y in zip(np.arange(0, 1, 1 / len(keys)),
                                             np.arange(0, 1, 1 / len(keys)))]
        for key, marker, color in zip(keys, markers, colors):
            data = j[key]
            plt.plot(data, marker, color=color,
                     markersize=4, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Val loss')
    plt.legend(fontsize='small')
    plt.savefig('images/train-process.pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def aaai_table():
    # retest these
    advae_json = 'images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_1.json'
    nodef_json = 'images/nodef-mnist.json'
    hgd_json = 'images/test-result-MNIST-mnistcnn-cnn1AE-B2.json'
    defgan_json = 'images/defgan2.json'
    itadv_json = 'images/test-result-MNIST-mnistcnn-identityAE-ItAdv.json'
    with open(nodef_json, 'r') as fp:
        nodef = json.load(fp)
    with open(advae_json, 'r') as fp:
        advae = json.load(fp)
    with open(hgd_json, 'r') as fp:
        hgd = json.load(fp)
    with open(defgan_json, 'r') as fp:
        defgan = json.load(fp)
    with open(itadv_json, 'r') as fp:
        itadv = json.load(fp)

    # adjust the format of defgan
    defgan['PGD'] = [defgan['PGD']] * 20
    defgan['PGD'][9] = defgan['PGD 0.38']
    defgan['PGD'][11] = defgan['PGD 0.46']
    defgan['FGSM'] = [defgan['FGSM']] * 20
    defgan['no atttack AE'] = defgan['CNN clean']
    defgan['Hop'] = [0] * 20
    rows = []
    data = [nodef, hgd, defgan, advae, itadv]
    # header
    rows.append(['attacks', 'no def', 'HGD', 'DefGAN', 'AdvAE', 'ItAdvTrain'])
    # data
    # FIXME atttack
    rows.append(['no attack'] + [d['no atttack AE'] for d in data])
    rows.append([r'FGSM $\ell_\infty$'] + [d['FGSM'][7] for d in data])
    rows.append([r'PGD $\ell_\infty, \epsilon=0.3$'] + [d['PGD'][7] for d in data])
    rows.append([r'PGD $\ell_\infty, \epsilon=0.38$'] + [d['PGD'][9] for d in data])
    rows.append([r'PGD $\ell_\infty, \epsilon=0.46$'] + [d['PGD'][11] for d in data])
    rows.append(['HSJA (black) $\epsilon=0.3$'] + [d['Hop'][7] for d in data])
    rows.append(['HSJA (black) $\epsilon=0.38$'] + [d['Hop'][9] for d in data])
    rows.append(['HSJA (black) $\epsilon=0.46$'] + [d['Hop'][11] for d in data])
    rows.append([r'CW $\ell_2$'] + [d['CW'] for d in data])

    for row in rows[1:]:
        print(row[0], end=' & ')
        for i in row[1:]:
            print('{:.1f}'.format(i * 100), end=' & ')
        print('\b\b', end='')
        print(r'\\')
def aaai_table_cifar():
    # retest these
    advae_json = 'images/test-result-CIFAR10-resnet29-dunet-C0_A2_2.json'
    nodef_json = 'images/nodef-cifar.json'
    hgd_json = 'images/test-result-CIFAR10-resnet29-dunet-B2.json'
    itadv_json = 'images/test-result-CIFAR10-resnet29-identityAE-ItAdv.json'
    with open(nodef_json, 'r') as fp:
        nodef = json.load(fp)
    with open(advae_json, 'r') as fp:
        advae = json.load(fp)
    with open(hgd_json, 'r') as fp:
        hgd = json.load(fp)
    with open(itadv_json, 'r') as fp:
        itadv = json.load(fp)

    rows = []
    data = [nodef, hgd, advae, itadv]
    # header
    rows.append(['attacks', 'no def', 'HGD', 'AdvAE', 'ItAdv'])
    # data
    # FIXME atttack
    rows.append(['no attack'] + [d['no atttack AE'] for d in data])
    rows.append([r'FGSM $\ell_\infty$'] + [d['FGSM'][3] for d in data])
    rows.append([r'PGD $\epsilon=8/255$'] + [d['PGD'][3] for d in data])
    rows.append([r'PGD $\epsilon=14/255$'] + [d['PGD'][6] for d in data])
    rows.append([r'PGD $\epsilon=20/255$'] + [d['PGD'][9] for d in data])
    rows.append([r'CW $\ell_2$'] + [d['CW'] for d in data])
    rows.append(['HSJA $\epsilon=8/255$'] + [d['Hop'][3] for d in data])
    rows.append(['HSJA $\epsilon=14/255$'] + [d['Hop'][6] for d in data])
    rows.append(['HSJA $\epsilon=20/255$'] + [d['Hop'][9] for d in data])

    for row in rows[1:]:
        print(row[0], end=' & ')
        for i in row[1:]:
            print('{:.1f}'.format(i * 100), end=' & ')
        print('\b\b', end='')
        print(r'\\')
def aaai_table_transfer():
    jsons = ['images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_1.json',
             # 'images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_1-TO-DefenseGAN_a.json',
             # 'images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_1-TO-DefenseGAN_b.json',
             # 'images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_1-TO-DefenseGAN_c.json',
             # 'images/test-result-MNIST-mnistcnn-cnn1AE-C0_A2_1-TO-DefenseGAN_d.json',
             # 'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-cnn1AE-C0_A2_1-ENSEMBLE-DefenseGAN_a.json',
             # 'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-cnn1AE-C0_A2_1-ENSEMBLE-DefenseGAN_b.json',
             # 'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-cnn1AE-C0_A2_1-ENSEMBLE-DefenseGAN_c.json',
             # 'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b]-cnn1AE-C0_A2_1-ENSEMBLE-DefenseGAN_d.json',
             'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b-DefenseGAN_c-DefenseGAN_d]-cnn1AE-C0_A2_1-ENSEMBLE-DefenseGAN_a.json',
             'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b-DefenseGAN_c-DefenseGAN_d]-cnn1AE-C0_A2_1-ENSEMBLE-DefenseGAN_b.json',
             'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b-DefenseGAN_c-DefenseGAN_d]-cnn1AE-C0_A2_1-ENSEMBLE-DefenseGAN_c.json',
             'images/test-result-MNIST-[mnistcnn-DefenseGAN_a-DefenseGAN_b-DefenseGAN_c-DefenseGAN_d]-cnn1AE-C0_A2_1-ENSEMBLE-DefenseGAN_d.json'
             
    ]
    data = []
    for fname in jsons:
        with open(fname, 'r') as fp:
            j = json.load(fp)
            data.append(j)
    rows = []
    # header
    rows.append(['attacks', 'X/X', 'X/A', 'X/B', 'X/C', 'X/D'])
    # data
    # FIXME atttack
    rows.append(['no attack'] + [d['no atttack AE'] for d in data])
    rows.append([r'FGSM $\ell_\infty$'] + [d['FGSM'][7] for d in data])
    rows.append([r'PGD $\ell_\infty$'] + [d['PGD'][7] for d in data])
    rows.append([r'CW $\ell_2$'] + [d['CW'] for d in data])
    # rows.append(['HSJA'] + [d['Hop'][7] for d in data])

    for row in rows[1:]:
        print(row[0], end=' & ')
        for i in row[1:]:
            print('{:.1f}'.format(i * 100), end=' & ')
        print('\b\b', end='')
        print(r'\\')

def aaai_plot():
    pass

def __test():
    # this is the main lambda plot
    plot_lambda_onto_epsilon()
    # plot_onto_lambda()
    plot_aesize_onto_epsilon()
    plot_defense_onto_epsilon()
    plot_defense_onto_epsilon_cifar()
    # plot_train_process()
    aaai_table()
    aaai_table_cifar()
    aaai_table_transfer()
