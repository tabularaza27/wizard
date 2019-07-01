import tensorflow as tf
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.signal import savgol_filter

### Plot Settings ###

PLOT_SIZE = (16, 9)
PLOT_STYLE = 'ggplot'

plt.rcParams["figure.figsize"] = PLOT_SIZE
plt.style.use(PLOT_STYLE)


### Loading Data Functionalities ###

def load_single_file(file_path):
    """Loads all summary from one tensorboard logfile and structures it in two dictionaries
    One for scalar values
    One for histograms

    The Dictionaries have following structure:

    {'summary_name': [np.array([step, value]), np.array([step, value])], 'summary_name': ....}

    Note:
        in our case we only have on step, value pair per summary per file

    Args:
        file_path (str): path to summary file

    Returns:
        dict: {'scalars': scalars_dict, 'histos': histos_dict}
    """
    scalars = dict()
    histos = dict()
    for index, summary in enumerate(tf.train.summary_iterator(file_path)):
        if summary.file_version:
            # first entry has info about file
            pass
        else:
            tag = summary.summary.value[0].tag
            histo_value = summary.summary.value[0].histo
            scalar_value = summary.summary.value[0].simple_value
            step = summary.step

            if histo_value.ByteSize() > 0:
                if tag not in histos.keys():
                    histos[tag] = [np.array([step, histo_value])]
                else:
                    histos[tag].append(np.array([step, histo_value]))
            else:
                if tag not in scalars.keys():
                    scalars[tag] = [np.array([step, scalar_value])]
                else:
                    scalars[tag].append(np.array([step, scalar_value]))

    return {'scalars': scalars, 'histos': histos}


def load_data(experiment_number, load_histos=False):
    """loads all data for given experiment number

    Args:
        experiment_number (int): number of experiment folder
        load_histos (bool): if True data for histos are also loaded

    Returns:
        dict: dictionary of all data for each player
    """

    # by default only load data for scalar values
    summary_types = ['scalars']
    if load_histos:
        summary_types.append('histos')

    log_path = '../logs/{}/*Agent*'.format(experiment_number)

    # get file paths for each player
    player_paths = glob.glob(log_path)
    player_files = {player_path.split('/')[-1]: glob.glob('{}/events*'.format(player_path)) for player_path in
                    player_paths}

    # for each player load and combine all log files
    player_data = {}
    for player_name, player_files in player_files.items():
        player_dict = {f'{summary_type}': {} for summary_type in summary_types}
        for path in player_files:
            file_dict = load_single_file(path)
            for summary_type in summary_types:
                for tag, value in file_dict[summary_type].items():
                    # print(value)
                    if tag not in player_dict[summary_type].keys():
                        player_dict[summary_type][tag] = value
                    else:
                        player_dict[summary_type][tag] += value

        player_data[player_name] = player_dict

        # sort summaries by time step
        for summary_type in summary_types:
            for tag, value in player_data[player_name][summary_type].items():
                player_data[player_name][summary_type][tag] = sorted(value, key=itemgetter(0))

    return player_data


def print_infos(data_set):
    """prints info about available agents and their summaries"""
    print('##### Agents #####')
    for agent in data_set.keys():
        print(agent)

    print('\n')
    print('##### Summaries #####')
    print('\n')
    for agent, agent_dict in data_set.items():
        print(f'### {agent} ###')
        print('\n')
        for summary_type, summaries in agent_dict.items():
            print(f'## {summary_type} ##')
            print('\n')
            for summary, data in summaries.items():
                print(summary, f'({len(data)} time steps)')
            print('\n')


### Plotting Functionalities ###

def smooth_data(data):
    """return smoothed data set (smoothing is done with savitzky_golay filter )

    It uses least squares to regress a small window of your data onto a polynomial,
    then uses the polynomial to estimate the point in the center of the window.

    Args:
        data (list): list of numbers (int, float)

    Return
        list: smoothed data
    """

    return


def plot_scalar(data, summary_name, agents=None, plot_original=True, smoothing=False, save_plot=True, plot_title=None,
                plot_name=None):
    """plotting scalar metrics for tensorboard summaries and save plot to /plots

    Args:
        data (dict): data containing all summary data (return of load_data())
        summary_name (str): name of summary to plot. run print_info(data) to see which summaries are available
        agents (list): list of agents to plot. If None all agents in data will be plotted
        plot_original (bool): If True, plot the raw data
        smoothing (bool): If True, plot the smoothed data
        save_plot (bool): If True, save plot to file
        plot_title (str): title of the plot. If None, summary_name is the title
        plot_name (str): file name when plot is saved. If None, use plot_title for naming the file
    """
    if not plot_title:
        plot_title = summary_name

    if not plot_name:
        plot_name = plot_title

    # agents for which data should be plotted
    if not agents:
        agents = data.keys()
    else:
        # check if provided agents exist
        for agent in agents:
            assert agent in data.keys(), f'Agent `{agent}` does not exist'

    agents_with_data = [agent for agent in agents if summary_name in data[agent]['scalars'].keys()]
    print(f'Agents that have data for metric `{summary_name}`: ', agents_with_data)

    # plotting
    fig, ax = plt.subplots()

    labels = []
    for index, agent in enumerate(agents_with_data):
        agent_data = data[agent]['scalars'][summary_name]
        x = [x[0] for x in agent_data]
        y = [x[1] for x in agent_data]

        if smoothing:
            # smooth data with savitzky_golay filter (https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
            # window size and polyorder can be adjusted or could be extracted as paramters
            smoothed_data = savgol_filter(y, window_length=31, polyorder=4)
            ax.plot(x, smoothed_data, color=f'C{index}')
            labels.append(f'{agent} smoothed')

        if plot_original:
            if smoothing:
                linestyle = '--'
                opacity = 0.5
            else:
                linestyle = '-'
                opacity = 1

            ax.plot(x, y, color=f'C{index}', linestyle=linestyle, alpha=opacity)
            labels.append(agent)

    ax.legend(labels)
    ax.set_title(plot_title)
    plt.show()

    if save_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        fig.savefig(f'plots/{plot_name}.png', transparent=False, dpi=300)
