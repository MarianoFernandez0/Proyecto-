import os
import PySimpleGUI as sg

current_path = os.getcwd()
os.makedirs(os.path.join(current_path, 'input'), exist_ok=True)
os.makedirs(os.path.join(current_path, 'output'), exist_ok=True)

SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'


def display_input_gui():
    sg.theme('Dark Blue 3')

    # advanced_section_esp = [[sg.Text('Parámetros')],
    #                         [sg.Text(' detección:          '),
    #                          sg.InputText('1', key='detection_algorithm', size=(7, 1))],
    #                         [sg.Text(' seguimiento:        '), sg.InputText('5', key='mtt_algorithm', size=(7, 1))],
    #                         [sg.Text(' PG:                 '), sg.InputText('0.997', key='PG', size=(7, 1))],
    #                         [sg.Text(' PD:                 '), sg.InputText('0.999', key='PD', size=(7, 1))],
    #                         [sg.Text(' gv:                 '), sg.InputText('50', key='gv', size=(7, 1))]]
    # #
    # advanced_section_en = [[sg.Text('Algorithm params')],
    #                        [sg.Text(' detection algorithm: '),
    #                         sg.InputText('Fluorescence', key='detection_algorithm', size=(7, 1))],
    #                        [sg.Text(' mtt_algorithm:       '), sg.InputText('5', key='mtt_algorithm', size=(7, 1))],
    #                        [sg.Text(' PG:                  '), sg.InputText('0.997', key='PG', size=(7, 1))],
    #                        [sg.Text(' PD:                  '), sg.InputText('0.999', key='PD', size=(7, 1))],
    #                        [sg.Text(' gv:                  '), sg.InputText('50', key='gv', size=(7, 1))]]
    advanced_section_en = [[sg.Text('Algorithm params')],
                           [sg.Text(' detection algorithm: '),
                            sg.Combo(values=['Fluorescence', 'Brightfield', 'Octave'], key='detection_algorithm',
                                     default_value='Fluorescence')],
                           [sg.Text(' mtt_algorithm:       '),
                            sg.Combo(values=['NN', 'GNN', 'PDAF', 'JPDAF', 'ENNJPDAF'], key='mtt_algorithm',
                                     default_value='ENNJPDAF')],
                           [sg.Text(' PG:                  '), sg.InputText('0.997', key='PG', size=(7, 1))],
                           [sg.Text(' PD:                  '), sg.InputText('0.999', key='PD', size=(7, 1))],
                           [sg.Text(' gv:                  '), sg.InputText('100', key='gv', size=(7, 1))]]
    #
    # esp_section = [
    #     [sg.Text('TDE', font='Courier 25'),
    #      sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''),
    #      sg.Text(''), sg.Text(''), sg.Button('EN', size=(1, 1), k='-CHANGE LAN-')],
    #     [sg.Text('')],
    #     [sg.Text('Entrada')],
    #     [sg.FileBrowse('secuencia de video', key='video_input', initial_folder=os.path.join(current_path, 'input'))],
    #     [sg.Text('  fps:                 '), sg.InputText('15', key='fps', size=(7, 1))],
    #     [sg.Text('  px2um:               '), sg.InputText('', key='px2um', size=(7, 1))],
    #     [sg.Text('')],
    #     [sg.T(SYMBOL_UP, enable_events=True, k='-OPEN ADVANCED-', text_color='black'),
    #      sg.T('Avanzado', enable_events=True, text_color='black', k='-OPEN ADVANCED-TEXT')],
    #     [sg.pin(sg.Column(advanced_section_esp, key='-ADV_SEC-', visible=False))],
    #     [sg.Button('Ok'), sg.Button('Cancelar')]
    # ]
    #
    # en_section = [
    #     [sg.Text('TDE', font='Courier 25'),
    #      sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''),
    #      sg.Text(''), sg.Button('ESP', size=(1, 1), k='-CHANGE LAN-')],
    #     [sg.Text('')],
    #     [sg.Text('Input')],
    #     [sg.FileBrowse('video sequence', key='video_input', initial_folder=os.path.join(current_path, 'input'))],
    #     [sg.Text('  fps:                 '), sg.InputText('15', key='fps', size=(7, 1))],
    #     [sg.Text('  px2um:               '), sg.InputText('0.1', key='px2um', size=(7, 1))],
    #     [sg.Text('')],
    #     [sg.T(SYMBOL_UP, enable_events=True, k='-OPEN ADVANCED-', text_color='black'),
    #      sg.T('Advanced', enable_events=True, text_color='black', k='-OPEN ADVANCED-TEXT')],
    #     [sg.pin(sg.Column(advanced_section_en, key='-ADV_SEC-', visible=False))],
    #     [sg.Button('Ok'), sg.Button('Cancel')]
    # ]
    #
    # layout = [[sg.pin(sg.Column(esp_section, key='-ESP_SEC-', visible=True))],
    #           [sg.pin(sg.Column(en_section, key='-EN_SEC-', visible=False))]]
    en_section = [
        [sg.Text('TDE', font='Courier 25')],
        [sg.Text('')],
        [sg.Text('Input')],
        [sg.FileBrowse('video sequence', key='video_input', initial_folder=os.path.join(current_path, 'input'))],
        [sg.Text('  fps:                 '), sg.InputText('15', key='fps', size=(7, 1))],
        [sg.Text('  px2um:               '), sg.InputText('', key='px2um', size=(7, 1))],
        [sg.Text('')],
        [sg.T(SYMBOL_UP, enable_events=True, k='-OPEN ADVANCED-', text_color='black'),
         sg.T('Advanced', enable_events=True, text_color='black', k='-OPEN ADVANCED-TEXT')],
        [sg.pin(sg.Column(advanced_section_en, key='-ADV_SEC-', visible=False))],
        [sg.Button('Ok'), sg.Button('Cancel')]
    ]

    layout = [[sg.pin(sg.Column(en_section, key='-EN_SEC-', visible=True))]]
    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)

    opened = False
    # esp_opened = False
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel', 'Cancelar', 'Ok'):
            break
        if event.startswith('-OPEN ADVANCED-'):
            opened = not opened
            window['-OPEN ADVANCED-'].update(SYMBOL_DOWN if opened else SYMBOL_UP)
            window['-ADV_SEC-'].update(visible=opened)
        # if event.startswith('-CHANGE LAN-'):
        #     esp_opened = not esp_opened
        #     window['-ESP_SEC-'].update(visible=esp_opened)
        #     window['-EN_SEC-'].update(visible=not esp_opened)

    window.close()
    return event, values


def save_detections_gui():
    sg.theme('Dark Blue 3')

    layout = [[sg.Text('TDE', font='Courier 25')],
              [sg.Text('')],
              [sg.FileSaveAs(button_text='Save detections file', key='detections_csv', initial_folder=current_path,
                             enable_events=True)]]

    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == 'detections_csv':
            break

    window.close()
    return values['detections_csv']


def save_tracks_gui(initial_folder):
    sg.theme('Dark Blue 3')

    layout = [[sg.Text('TDE', font='Courier 25')],
              [sg.Text('')],
              [sg.FileSaveAs(button_text='Save trajectories file', key='tracks_csv', initial_folder=initial_folder,
                             enable_events=True)]]

    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == 'tracks_csv':
            break

    window.close()
    return values['tracks_csv']


def save_vid_gui(tracks, initial_folder):
    sg.theme('Dark Blue 3')
    num_tracks = len(tracks['id'].unique())

    layout = [
        [sg.Text('TDE', font='Courier 25')],
        [sg.Text('')],
        [sg.Text('Results', font='Courier 10')],
        [sg.Text('  {} trajectories detected.'.format(num_tracks), font='Courier 9')],
        [sg.FileSaveAs(button_text='Save video', key='tracks_video', initial_folder=initial_folder, enable_events=True)],
        [sg.Button('Close')]
    ]

    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Close'):
            break

    window.close()
    return values['tracks_video']
