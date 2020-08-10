import PySimpleGUI as sg
import os
current_path = os.getcwd()
os.makedirs(os.path.join(current_path, 'input'), exist_ok=True)
os.makedirs(os.path.join(current_path, 'output'), exist_ok=True)


def display_gui():
    sg.theme('Dark Blue 3')   # Add a touch of color
    # All the stuff inside your window.
    layout = [
        [sg.Text('          TDE', font='Courier 25')],
        [sg.Text('')],
        [sg.Text('')],
        [sg.Text('Save video?           '), sg.Checkbox(' check', default=True, key='save_vid')],
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
        [sg.Text('Algorithm params (advanced)')],
        [sg.Text('  detection_algorithm:'), sg.InputText('1', key='detection_algorithm')],
        [sg.Text('  mtt_algorithm:      '), sg.InputText('5', key='mtt_algorithm')],
        [sg.Text('  PG:                 '), sg.InputText('0.997', key='PG')],
        [sg.Text('  PD:                 '), sg.InputText('0.999', key='PD')],
        [sg.Text('  gv:                 '), sg.InputText('50', key='gv')],

        [sg.Button('Ok'), sg.Button('Cancel')]
    ]

    # Create the Window
    window = sg.Window('Window Title', layout)
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel' or event == 'Ok':  # if user closes window or clicks cancel
            break
        print('You entered ', values)

    window.close()
    return event, values
