import PySimpleGUI as sg
import os

current_path = os.getcwd()
os.makedirs(os.path.join(current_path, 'input'), exist_ok=True)
os.makedirs(os.path.join(current_path, 'output'), exist_ok=True)

SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'


def display_input_gui():
    sg.theme('Dark Blue 3')  # Add a touch of color

    advanced_section_esp = [[sg.Text('Parámetros')],
                            [sg.Text(' detección:          '),
                             sg.InputText('1', key='detection_algorithm', size=(7, 1))],
                            [sg.Text(' seguimiento:        '), sg.InputText('5', key='mtt_algorithm', size=(7, 1))],
                            [sg.Text(' PG:                 '), sg.InputText('0.997', key='PG', size=(7, 1))],
                            [sg.Text(' PD:                 '), sg.InputText('0.999', key='PD', size=(7, 1))],
                            [sg.Text(' gv:                 '), sg.InputText('50', key='gv', size=(7, 1))]]

    advanced_section_en = [[sg.Text('Algorithm params')],
                           [sg.Text(' detection algorithm: '),
                            sg.InputText('1', key='detection_algorithm', size=(7, 1))],
                           [sg.Text(' mtt_algorithm:       '), sg.InputText('5', key='mtt_algorithm', size=(7, 1))],
                           [sg.Text(' PG:                  '), sg.InputText('0.997', key='PG', size=(7, 1))],
                           [sg.Text(' PD:                  '), sg.InputText('0.999', key='PD', size=(7, 1))],
                           [sg.Text(' gv:                  '), sg.InputText('50', key='gv', size=(7, 1))]]

    esp_section = [
        [sg.Text('TDE', font='Courier 25'),
         sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''),
         sg.Text(''), sg.Text(''), sg.Button('EN', size=(1, 1), k='-CHANGE LAN-')],
        [sg.Text('')],
        [sg.Text('')],
        [sg.Text('Guardar video?           '), sg.Checkbox('', default=True, key='save_vid')],
        [sg.Text('')],
        [sg.Text('Entrada')],
        [sg.Text('  secuencia:     '),
         sg.FileBrowse('', key='video_input', initial_folder=os.path.join(current_path, 'input'),
                       size=(4, 1))],
        [sg.Text('  fps:                 '), sg.InputText('15', key='fps', size=(7, 1))],
        [sg.Text('  px2um:               '), sg.InputText('', key='px2um', size=(7, 1))],
        [sg.Text('')],
        [sg.Text('Carpeta de resultados: '),
         sg.FolderBrowse('', key='output', initial_folder=os.path.join(current_path, 'output'), size=(4, 1))],
        [sg.Text('')],
        [sg.T(SYMBOL_UP, enable_events=True, k='-OPEN ADVANCED-', text_color='black'),
         sg.T('Avanzado', enable_events=True, text_color='black', k='-OPEN ADVANCED-TEXT')],
        [sg.pin(sg.Column(advanced_section_esp, key='-ADV_SEC-', visible=False))],
        [sg.Button('Ok'), sg.Button('Cancelar')]
    ]

    en_section = [
        [sg.Text('TDE', font='Courier 25'),
         sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''), sg.Text(''),
         sg.Text(''), sg.Button('ESP', size=(1, 1), k='-CHANGE LAN-')],
        [sg.Text('')],
        [sg.Text('')],
        [sg.Text('Save video?           '), sg.Checkbox('', default=True, key='save_vid')],
        [sg.Text('')],
        [sg.Text('Input')],
        [sg.Text('  sequence:      '),
         sg.FileBrowse('', key='video_input', initial_folder=os.path.join(current_path, 'input'),
                       size=(4, 1))],
        [sg.Text('  fps:                 '), sg.InputText('15', key='fps', size=(7, 1))],
        [sg.Text('  px2um:               '), sg.InputText('0.1', key='px2um', size=(7, 1))],
        [sg.Text('')],
        [sg.Text('Output Folder:         '),
         sg.FolderBrowse('', key='output', initial_folder=os.path.join(current_path, 'output'), size=(4, 1))],
        [sg.Text('')],
        [sg.T(SYMBOL_UP, enable_events=True, k='-OPEN ADVANCED-', text_color='black'),
         sg.T('Advanced', enable_events=True, text_color='black', k='-OPEN ADVANCED-TEXT')],
        [sg.pin(sg.Column(advanced_section_en, key='-ADV_SEC-', visible=False))],
        [sg.Button('Ok'), sg.Button('Cancel')]
    ]

    layout = [[sg.pin(sg.Column(esp_section, key='-ESP_SEC-', visible=True))],
              [sg.pin(sg.Column(en_section, key='-EN_SEC-', visible=False))]]
    # All the stuff inside your window.

    # Create the Window
    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)
    # Event Loop to process "events" and get the "values" of the inputs
    opened = False
    esp_opened = False
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel', 'Cancelar', 'Ok'):  # if user closes window or clicks cancel
            # print('You entered ', values)
            break
        if event.startswith('-OPEN ADVANCED-'):
            opened = not opened
            window['-OPEN ADVANCED-'].update(SYMBOL_DOWN if opened else SYMBOL_UP)
            window['-ADV_SEC-'].update(visible=opened)
        if event.startswith('-CHANGE LAN-'):
            esp_opened = not esp_opened
            window['-ESP_SEC-'].update(visible=esp_opened)
            window['-EN_SEC-'].update(visible=not esp_opened)

    window.close()
    return event, values


def display_results_gui(tracks):
    sg.theme('Dark Blue 3')  # Add a touch of color

    num_tracks = len(tracks['id'].unique())
    advanced_section = [[sg.Text('Results')],
                        [sg.Text('  Number of trajectories detected: {}'.format(num_tracks))]]

    # All the stuff inside your window.
    layout = [
        [sg.Text('TDE', font='Courier 25')],
        [sg.Text('')],
        [sg.Text('Results', font='Courier 10')],
        [sg.Text('  {} trajectories detected.'.format(num_tracks), font='Courier 9')],
        [sg.Button('Close')]
    ]

    # Create the Window
    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Close'):  # if user closes window or clicks cancel
            # print('You entered ', values)
            break

    window.close()
    return event, values
