def PrintWithBorders(info, border_char='-', width=70):
    """
    This function is used for printing information with border
    """
    border = border_char * width
    print(border)
    content_width = width - 4
    words = info.split()
    line = ''
    
    for word in words:
        if len(line) + len(word) + 1 > content_width:
            print('| ' + line.ljust(content_width) + ' |')
            line = word
        else:
            line += (' ' + word if line else word)
            
    if line:
        print('| ' + line.ljust(content_width) + ' |')
    
    print(border)
