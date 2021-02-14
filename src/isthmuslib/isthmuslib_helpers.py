#########################
## version
#########################

def version():
    return '0.0.4'


##############################
## Misc useful stuff
##############################

def handystrings(whichone='help', toconsole=True):
    handydict = {
        "pandas width": "pd.set_option('display.max_colwidth', None)",
        "parent directory": "sys.path.append(os.path.abspath('../'))",
        "disable scrolling": ''
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

def demodata():
    x = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    y = [1, 4, 8, 20, 30, 40, 120, 250, 700, 1100]
    return x, y
