import PySimpleGUI as sg
import os

current_path = os.getcwd()
os.makedirs(os.path.join(current_path, 'input'), exist_ok=True)
os.makedirs(os.path.join(current_path, 'output'), exist_ok=True)

SYMBOL_UP = '▲'
SYMBOL_DOWN = '▼'


def display_gui():
    sg.theme('Dark Blue 3')  # Add a touch of color

    advanced_section = [[sg.Text('Algorithm params')],
                        [sg.Text('  detection_algorithm:'), sg.InputText('1', key='detection_algorithm')],
                        [sg.Text('  mtt_algorithm:      '), sg.InputText('5', key='mtt_algorithm')],
                        [sg.Text('  PG:                 '), sg.InputText('0.997', key='PG')],
                        [sg.Text('  PD:                 '), sg.InputText('0.999', key='PD')],
                        [sg.Text('  gv:                 '), sg.InputText('50', key='gv')]]

    # All the stuff inside your window.
    layout = [
        [sg.Text('          TDE', font='Courier 25')],
        [sg.Text('')],
        [sg.Text('')],
        [sg.Text('Save video?           '), sg.Checkbox('', default=True, key='save_vid')],
        [sg.Text('')],
        [sg.Text('Input')],
        [sg.Text('  tif_video_input:    '),
         sg.FileBrowse('-.tif', key='tif_video_input', initial_folder=os.path.join(current_path, 'input'))],
        [sg.Text('  fps:                '), sg.InputText('15', key='fps')],
        [sg.Text('  px2um:              '), sg.InputText('0.1', key='px2um')],
        [sg.Text('  ROIx:               '), sg.InputText('512', key='ROIx')],
        [sg.Text('  ROIy:               '), sg.InputText('512', key='ROIy')],
        [sg.Text('')],
        [sg.Text('Output Folder:        '),
         sg.FolderBrowse('-', key='output', initial_folder=os.path.join(current_path, 'output'))],
        [sg.Text('')],
        [sg.T(SYMBOL_UP, enable_events=True, k='-OPEN ADVANCED-', text_color='black'),
         sg.T('Advanced', enable_events=True, text_color='black', k='-OPEN ADVANCED-TEXT')],
        [sg.pin(sg.Column(advanced_section, key='-ADV_SEC-', visible=False))],
        [sg.Button('Ok'), sg.Button('Cancel')]
    ]

    # Create the Window
    window = sg.Window('Tracking de Espermatozoides', layout, no_titlebar=False, alpha_channel=1, grab_anywhere=True)
    # Event Loop to process "events" and get the "values" of the inputs
    opened = False
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel', 'Ok'):  # if user closes window or clicks cancel
            print('You entered ', values)
            break
        if event.startswith('-OPEN ADVANCED-'):
            opened = not opened
            window['-OPEN ADVANCED-'].update(SYMBOL_DOWN if opened else SYMBOL_UP)
            window['-ADV_SEC-'].update(visible=opened)

    window.close()
    return event, values
