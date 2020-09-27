import os
import PySimpleGUI as sg

current_path = os.getcwd()
os.makedirs(os.path.join(current_path, 'input'), exist_ok=True)
os.makedirs(os.path.join(current_path, 'output'), exist_ok=True)

SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'


def display_input_gui():
    sg.theme('Dark Blue 3')

    advanced_section = [[sg.Text('Algorithm params')],
                           [sg.Text(' detection algorithm: '),
                            sg.Combo(values=['Fluorescence', 'Brightfield', 'Octave'], key='detection_algorithm',
                                     default_value='Fluorescence')],
                           [sg.Text(' mtt_algorithm:       '),
                            sg.Combo(values=['NN', 'GNN', 'PDAF', 'JPDAF', 'ENNJPDAF'], key='mtt_algorithm',
                                     default_value='ENNJPDAF')],
                           [sg.Text(' PG:                  '), sg.InputText('0.997', key='PG', size=(7, 1))],
                           [sg.Text(' PD:                  '), sg.InputText('0.999', key='PD', size=(7, 1))],
                           [sg.Text(' gv:                  '), sg.InputText('100', key='gv', size=(7, 1))]]

    section = [
        [sg.Text('TDE', font='Courier 25')],
        [sg.Text('')],
        [sg.Text('Input')],
        [sg.FileBrowse('video sequence', key='video_input', initial_folder=os.path.join(current_path, 'input'))],
        [sg.Text('  fps:              '), sg.InputText('15', key='fps', size=(7, 1))],
        [sg.Text('  um_per_px:        '), sg.InputText('', key='um_per_px', size=(7, 1))],
        [sg.FolderBrowse('output folder', key='out_dir', initial_folder=os.path.join(current_path))],
        [sg.Text('')],
        [sg.T(SYMBOL_UP, enable_events=True, k='-OPEN ADVANCED-', text_color='black'),
         sg.T('Advanced', enable_events=True, text_color='black', k='-OPEN ADVANCED-TEXT')],
        [sg.pin(sg.Column(advanced_section, key='-ADV_SEC-', visible=False))],
        [sg.Button('Ok'), sg.Button('Cancel')]
    ]

    layout = [[sg.pin(sg.Column(section, key='-EN_SEC-', visible=True))]]
    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)

    opened = False
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel', 'Cancelar', 'Ok'):
            break
        if event.startswith('-OPEN ADVANCED-'):
            opened = not opened
            window['-OPEN ADVANCED-'].update(SYMBOL_DOWN if opened else SYMBOL_UP)
            window['-ADV_SEC-'].update(visible=opened)
    window.close()
    return event, values


def progress_gui(text):
    sg.theme('Dark Blue 3')
    layout = [[sg.Text('TDE', font='Courier 25')],
              [sg.Text('')],
              [sg.Text(str(text))]]
    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)
    window.Finalize()
    return window


def drawing_vid_gui(tracks):
    sg.theme('Dark Blue 3')
    num_tracks = len(tracks['id'].unique())
    layout = [
        [sg.Text('TDE', font='Courier 25')],
        [sg.Text('')],
        [sg.Text('{} trajectories detected.'.format(num_tracks), font='Courier 10')],
        [sg.Text('Saving video...')]]
    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)
    window.Finalize()
    return window
