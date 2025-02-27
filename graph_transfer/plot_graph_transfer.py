import pandas
from matplotlib import pyplot as plt
import matplotlib 


plot= {'line': {'3': {'adgn': 1.1762166e-06, 'swan': 5.215618e-07, 'gat': 0.0001464392209999, 'gcn': 0.000166354495, 'gin': 6.923546100000001e-06, 'gps': 1.6044336775e-07, 'sage': 1.5305805500000002e-06}, '5': {'adgn': 2.94894e-06, 'swan': 4.4123672e-07, 'gat': 0.00701326495, 'gcn': 0.00364969595, 'gin': 0.0003825089324999, 'gps': 3.54280785e-06, 'sage': 0.0001612488175}, '10': {'adgn': 1.594517e-06, 'swan': 1.6841603e-06, 'gat': 0.01249609625, 'gcn': 0.0122033219999999, 'gin': 0.00975612675, 'gps': 7.553347550000001e-05, 'sage': 0.001292575125}, '50': {'adgn': 1.362915e-05, 'swan': 3.5678338e-06, 'gat': 0.03193722575, 'gcn': 0.030064674, 'gin': 0.02958572, 'gps': 0.06015534957125, 'sage': 0.0320354755}}, 'ring': {'3': {'adgn': 4.4347144e-05, 'swan': 2.7112956e-05, 'gat': 0.008877214875, 'gcn': 0.00457172, 'gin': 0.00125871221, 'gps': 0.0019136149, 'sage': 5.8633889e-05}, '5': {'adgn': 0.0010127842, 'swan': 0.00027417045, 'gat': 0.0118532502499999, 'gcn': 0.0165461825, 'gin': 0.0082046734999999, 'gps': 0.0009866837, 'sage': 9.543101e-05}, '10': {'adgn': 0.0041048015, 'swan': 0.0043941233, 'gat': 0.014097925, 'gcn': 0.04415201825, 'gin': 0.0399007055, 'gps': 4.774982375e-05, 'sage': 0.001774213375}, '50': {'adgn': 0.008844013, 'swan': 0.0053357505, 'gat': 0.02654469, 'gcn': 0.02653098575, 'gin': 0.0263560615, 'gps': 0.03321089015, 'sage': 0.02384170275}}, 'crossed-ring': {'3': {'adgn': 3.4476772e-05, 'swan': 5.829515e-07, 'gat': 0.0078059775999999, 'gcn': 0.01727941375, 'gin': 0.00255819885, 'gps': 3.35535525e-06, 'sage': 4.29032825e-05}, '5': {'adgn': 9.884677e-05, 'swan': 1.0929035e-06, 'gat': 0.01434588875, 'gcn': 0.01676719325, 'gin': 0.01288108725, 'gps': 6.980356525e-06, 'sage': 0.0001526801775}, '10': {'adgn': 3.1738877e-05, 'swan': 1.4969074e-06, 'gat': 0.0182240924999999, 'gcn': 0.03412708625, 'gin': 0.032688238, 'gps': 4.72305175e-05, 'sage': 0.0001699585925}, '50': {'adgn': 0.00011774405, 'swan': 1.3560239e-05, 'gat': 0.02659736775, 'gcn': 0.02619291125, 'gin': 0.0258902674999999, 'gps': 0.0600447415775, 'sage': 0.026597337}}}
plot_std = {'line': {'3': {'gat': 7.3803111461609e-05, 'gcn': 5.235153227648037e-05, 'gin': 2.2850258002001142e-06, 'adgn': 0.0, 'swan': 0.0, 'gps': 1.323679901585143e-07, 'sage': 2.8587782896823253e-07}, '5': {'gat': 0.001426542600667, 'gcn': 0.0006011976495882, 'gin': 0.0001253129469175, 'adgn': 0.0, 'swan': 0.0, 'gps': 2.7863387864046787e-06, 'sage': 4.274564114438833e-05}, '10': {'gat': 0.0005760125720314, 'gcn': 0.0001754771481589, 'gin': 0.0007302258132666, 'adgn': 0.0, 'swan': 0.0, 'gps': 2.836822703853151e-05, 'sage': 0.0001569409344999}, '50': {'gat': 0.0001536838790686, 'gcn': 4.266310431743056e-05, 'gin': 8.475452675816187e-05, 'adgn': 0.0, 'swan': 0.0, 'gps': 0.0418082753987453, 'sage': 3.2912926032182737e-06}}, 'ring': {'3': {'gat': 0.0017374272669967, 'gcn': 0.000443426535539, 'gin': 0.000390898491277, 'adgn': 0.0, 'swan': 0.0, 'gps': 0.0017421471242796, 'sage': 2.07975184853063e-05}, '5': {'gat': 0.001191047727559, 'gcn': 0.0011271628043973, 'gin': 0.0021888817647125, 'adgn': 0.0, 'swan': 0.0, 'gps': 0.0002621496698991, 'sage': 1.7772508473077105e-05}, '10': {'gat': 0.0008380612051129, 'gcn': 0.0025351119521773, 'gin': 0.0053259970811451, 'adgn': 0.0, 'swan': 0.0, 'gps': 1.3920278168205652e-05, 'sage': 0.0014528444671035}, '50': {'gat': 9.99881037023904e-05, 'gcn': 6.838209103449473e-05, 'gin': 0.0002863729240652, 'adgn': 0.0, 'swan': 0.0, 'gps': 0.0321153123041869, 'sage': 0.0032139805135677}}, 'crossed-ring': {'3': {'gat': 0.0024622432714702, 'gcn': 0.0018720658577443, 'gin': 0.0006738694072134, 'adgn': 0.0, 'swan': 0.0, 'gps': 2.414949514683156e-06, 'sage': 1.8445254189435607e-05}, '5': {'gat': 0.0004554453566875, 'gcn': 0.0007896913641397, 'gin': 0.0028679138200762, 'adgn': 0.0, 'swan': 0.0, 'gps': 4.432031258357929e-06, 'sage': 2.280098599129049e-05}, '10': {'gat': 0.0008483434676644, 'gcn': 0.0017729676212919, 'gin': 0.0026287296429217, 'adgn': 0.0, 'swan': 0.0, 'gps': 5.937102262869011e-06, 'sage': 5.518878363604774e-05}, '50': {'gat': 7.088194410414219e-08, 'gcn': 1.1894885325915745e-05, 'gin': 0.0001150675220598, 'adgn': 0.0, 'swan': 0.0, 'gps': 0.0401599854823088, 'sage': 1.0395191195927085e-07}}}
for data in ['line', 'ring', 'crossed-ring']: 
    plot[data] = pandas.DataFrame(plot[data])
    plot_std[data] = pandas.DataFrame(plot_std[data])

markers = {
    'gin': 'o',
    'gcn': 's',
    'gat': '^',
    'sage': 'v',
    'swan': 'd',
    'adgn': 'p',
    'gps': 'x'
}

mapper={
    'gin': 'gin'.upper(),
    'gcn': 'gcn'.upper(),
    'gat': 'gat'.upper(),
    'sage': 'sage'.upper(),
    'swan': 'swan'.upper(),
    'adgn': 'adgn'.upper(),
    'gps': 'gps'.upper()
}

seaborn_colorblind = {
    'gcn': '#949494', 
    'gat': '#de8f05', 
    'gin': '#d55e00',
    'sage': '#cc78bc', 
    'gps': '#FFD700', 
    'adgn': '#0173b2', 
    'swan': '#029e73'
}

matplotlib.rcParams.update({'font.size': 22})
colors = seaborn_colorblind

fig, ax =plt.subplots(figsize=(20,4.5), ncols=3)
for i, gtype in enumerate(['line', 'ring', 'crossed-ring']): 
    df = pandas.DataFrame(plot[gtype])
    df_s = pandas.DataFrame(plot_std[gtype])
    x = [int(c) for c in df.columns]
    for k in mapper.keys():
        label = None
        if i == 0:
            label = mapper[k]
        ax[i].plot(x, df.loc[k].values, 
                    marker=markers[k], 
                    color=colors[k], 
                    markersize=9,#7, 
                    linewidth=2.0, 
                    linestyle='solid',
                    label=label)
        ax[i].fill_between(x, df.loc[k].values-df_s.loc[k].values, df.loc[k].values+df_s.loc[k].values, 
                            color=colors[k], 
                            alpha=0.2, 
                            linewidth=2.0)
    ax[i].set_xticks(x)
    ax[i].set(
        xlabel='Source-Target distance (#hops)',
        ylabel='Mean Squared Error' if i ==0 else ''
    )
    ax[i].grid('on')
    ax[i].set_yscale('log')

lgd=fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10),
        ncol=5, fancybox=True, shadow=True)

plt.tight_layout()
fig.tight_layout()
fig.savefig(f'GraphTransfer_logscale_all.png', bbox_inches='tight')
plt.close()
    
