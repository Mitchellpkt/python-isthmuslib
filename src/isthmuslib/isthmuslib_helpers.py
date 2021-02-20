#########################
## version
#########################

def version():
    return '0.0.11'


##############################
## Misc useful stuff
##############################

def handystrings(whichone='help', toconsole=True):
    handydict = {
        "pandas width": "pd.set_option('display.max_colwidth', None)",
        "parent directory": "sys.path.append(os.path.abspath('../'))",
        "disable scrolling": '',
        "buildpackage": "python3 -m pip install --upgrade build && python3 -m build",
        "distributepackage": "python3 -m pip install --upgrade twine && python3 -m twine dist/*"
    }

    if whichone == 'help':
        response = str(handydict.keys())
    else:
        try:
            response = handydict[whichone]
        except:
            response = "entry not found, try handystrings('help')"
    if toconsole:
        print(response)
    return response


##############################
## Misc useful stuff
##############################

def demoData(setID=1):
    # Warning, the format of the data depends on the set ID/type
    if setID == 1:
        x = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        y = [1, 4, 8, 20, 30, 40, 120, 250, 700, 1100]
        return x, y

    if setID == 2:
        x1, y1, = demoData(1)
        color1 = 'r'
        label1 = 'Group data'

        y2, temp = demoData(1)
        x2 = [3, 3, 7, 30, 60, 75, 120, 250, 850, 900]
        color2 = '#3366FF'
        markersize2 = 300
        label2 = 'Control case'
        markerstyle2 = '^'

        data1 = {'x': x1, 'y': y1, 'markercolor': color1, 'label': label1}
        data2 = {'x': x2, 'y': y2, 'markercolor': color2, 'markersize': markersize2, 'label': label2,
                 'markerstyle': markerstyle2}

        return [data1, data2]
