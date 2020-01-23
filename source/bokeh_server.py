import os, sys, importlib
os.chdir('/home/jdehning/ownCloud/studium/two_photon_imaging/source')
sys.path.insert(0,'/home/jdehning/ownCloud/studium/two_photon_imaging/source')
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row, column, gridplot
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, CheckboxButtonGroup, RadioButtonGroup
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
from dask.distributed import Client
import dask.delayed
import allen_comparison
import time

#output_notebook()
directory = "/home/jdehning/tmp/Emx1-s_highzoom"
#directory = "/scratch.local/jdehning/calcium_ephys_comparison_data/processed_data/Emx1-s_highzoom"
#directory = "/scratch.local/jdehning/calcium_ephys_comparison_data/processed_data/Emx1-s_lowzoom"



#client = Client('localhost:8786')
client = Client('localhost:42747')

client.upload_file('allen_comparison.py')
client.run(importlib.import_module, 'allen_comparison')
futures = []
last_pos = 0


def modify_doc(doc):
    # Set up data
    #ephys, ophys, dt = allen_comparison.open_dir(directory)
    ephys = allen_comparison.open_ephys(directory, client)
    ophys = allen_comparison.open_ophys(directory, client)
    k_arr = np.arange(1, 35)
    sources = []
    plots = []
    for i in range(len(ephys)):
        source1 = ColumnDataSource(data=dict(x=k_arr, y=np.zeros_like(k_arr)))
        source2 = ColumnDataSource(data=dict(x=k_arr, y=np.zeros_like(k_arr)))
        source3 = ColumnDataSource(data=dict(x=k_arr, y=np.zeros_like(k_arr)))

        sources.append({'ephys_spikes': source1, 'ophys': source2, 'ephys_raw': source3})
        # Set up plot
        plot = figure(plot_height=170, plot_width=300,
                      tools="crosshair,pan,reset,save,wheel_zoom",
                      x_range=[min(k_arr), max(k_arr)], y_range=[-0.1, 1])
        plot.axis.visible = False
        plot.toolbar.logo = None
        plot.toolbar_location = None
        plot.xgrid.visible = False
        plot.ygrid.visible = False

        plot.line('x', 'y', source=source1, line_width=3, line_alpha=0.4)
        plot.line('x', 'y', source=source3, line_width=3, line_alpha=0.8)
        plot.line('x', 'y', source=source2, line_width=3, line_alpha=0.6, color='red')
        plots.append(plot)

    # Set up widgets
    # text = TextInput(title="title", value='my sine wave')
    tau = Slider(title="tau", value=1, start=1.5, end=2, step=0.001)
    subtract_neuropil = Slider(title="Subtract Neuropil", value=0, start=-1, end=2.0, step=0.001)
    use_spikes = CheckboxButtonGroup(labels=["Use spikes", "Use raw voltage"], active=[0])
    frequency = Slider(title="Frequency", value=0, start=0, end=1000, step=10)
    frequency_fine = Slider(title="Frequency Finetuning", value=2, start=0.05, end=20, step=0.05)
    rectify = RadioButtonGroup(labels=["No rectification", 'Absolute', 'Squared'], active=0)
    thresh_voltage = Slider(title="threshold_voltage", value=-10, start=-10, end=10, step=0.1)

    # phase = Slider(title="phase", value=0.0, start=0.0, end=2*np.pi)
    # freq = Slider(title="frequency", value=1.0, start=0.1, end=5.1, step=0.001)

    # Set up callbacks
    # def update_title(attrname, old, new):
    #    plot.title.text = text.value

    def replot(*args):
        global futures
        start_time = time.time()
        indices_finished = []
        for i, future in enumerate(futures):
            if future.done():
                indices_finished.append(i)
                res = future.result()
                # res = future._result()
                #print(res)
                for corr_dic in res:
                    print(corr_dic['index'], corr_dic['rec'])
                    sources[corr_dic['index']][corr_dic['rec']].data = dict(x=k_arr, y=corr_dic['corr'])
                # del future
        indices_finished.reverse()
        for i in indices_finished:
            futures.pop(i)
            # print('elapsed1: {:.1f} ms'.format((time.time()-start_time)*1000))

    # text.on_change('value', update_title)
    def start_comp(*args):
        global futures
        global last_pos
        # print(len(futures), futures[0])
        # Get the current slider values
        start_time = time.time()
        # print('elapsed1: {:.1f} ms'.format((time.time()-start_time)*1000))
        # print(futures[0])
        # futures = []
        t = tau.value
        subtr_fac = subtract_neuropil.value
        use_spks = use_spikes.active
        thrsh_v = thresh_voltage.value
        freq = frequency.value
        freq_f = frequency_fine.value
        rctfy = rectify.active
        if t + subtr_fac +sum(use_spks) +thrsh_v +freq+rctfy+freq_f == last_pos:
            return
        else:
            last_pos = t + subtr_fac +sum(use_spks) +thrsh_v +freq+rctfy+freq_f
        if len(futures) > 1:
            futures = [futures[-1]]

        # print('elapsed1: {:.1f} ms'.format((time.time()-start_time)*1000))
        to_compute = []
        to_compute.append(allen_comparison.make_calcium(ophys, k_arr, t, subtr_fac))
        if 0 in use_spks:
            to_compute.append(allen_comparison.make_spikes(ephys, k_arr, 0, freq+freq_f,rctfy,thrsh_v))
        if 1 in use_spks:
            to_compute.append(allen_comparison.make_spikes(ephys, k_arr, 1, freq+freq_f,rctfy,thrsh_v))


        @dask.delayed
        def concat(corr):
            return sum(corr, [])

        print('elapsed2: {:.1f} ms'.format((time.time() - start_time) * 1000))
        corr = client.compute(concat(to_compute))
        #
        futures.append(corr)
        print('elapsed3: {:.1f} ms'.format((time.time() - start_time) * 1000))
        # Generate the new curve
        # data = list(zip(*allen_comparison.make_corr_arr(client, data_allen,t)))
        # for i, (ephys, ophys) in enumerate(data):
        #    sources[i][0].data = dict(x=k_arr, y=ephys)
        #    sources[i][1].data = dict(x=k_arr, y=ophys*t)

    # for w in [tau]:
    #    w.on_change('value', start_comp)
    doc.add_periodic_callback(replot, 20)
    doc.add_periodic_callback(start_comp, 80)

    # Set up layouts and add to document
    inputs = column(tau, subtract_neuropil, use_spikes, thresh_voltage, frequency, frequency_fine, rectify)
    grid = gridplot(plots[:], ncols=4)

    doc.add_root(row(inputs, grid, width=1000))
    doc.title = "Sliders"


modify_doc(curdoc())