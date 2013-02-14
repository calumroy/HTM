
import pygame

def initialize_drawing():
    pygame.init()
def quit_drawing():
    pygame.quit()
def draw_HTM(HTM,input):
    numberLayers = len(HTM.HTMLayerArray)
    layer = 0
    drawing = True
    colors = [(255,255,255), (0,0,0),(100,100,100)]    # Set up colors [white, black, grey]
    c = len(HTM.HTMLayerArray[0].columns[0])         # This is an NxN chess board.
    r = len(HTM.HTMLayerArray[0].columns)                  # The number of rows in the screen
    surface_sz = 640        # Proposed physical surface size.
    sq_sz = surface_sz // c    # sq_sz is length of a square.
    surface_sz = c * sq_sz     # Adjust to exactly fit n squares.
    font = pygame.font.Font(None, 36)
    # Create the surface of (width, height)
    surface = pygame.display.set_mode((surface_sz, r*sq_sz))
    # Use an extra offset to centre the text in its square.
    # If the square is too small, offset becomes negative,
    #   but it will still be centered :-)
    #offset = (sq_sz) // 2
    offset = 0.0
    # REQUIRES A CLEANUP SHOULD NOT BE DONE THIS WAY
    # Display the HTM for the first time 
    for row in range(r):           # Draw each row of the board
            for col in range(c):       # Run through cols drawing squares
                the_square = (col*sq_sz, row*sq_sz, sq_sz, sq_sz)
                if HTM.HTMLayerArray[0].columns[row][col].activeState==True:
                    surface.fill(colors[0], the_square)
                else:
                    surface.fill(colors[1], the_square)
                text = font.render("%s" % HTM.HTMLayerArray[0].columns[row][col].overlap, 1, (255, 50, 0))
                textpos = (col*sq_sz+offset,row*sq_sz+offset)
                surface.blit(text,textpos)
    while drawing == True:
        # Look for an event from keyboard, mouse, etc.
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                surface.fill(colors[2])
                mouse_xy = pygame.mouse.get_pos()
                pos_x = mouse_xy[0]/sq_sz
                pos_y = mouse_xy[1]/sq_sz
                print "(x,y) = %s, %s"%(pos_x,pos_y)
                #print len(HTM.columns[pos_x][pos_y].potentialSynapses)
                for s in HTM.HTMLayerArray[layer].columns[pos_y][pos_x].potentialSynapses:
                    the_square = (s.pos_x*sq_sz, s.pos_y*sq_sz, sq_sz, sq_sz)
                    if s.sourceInput==1:
                        surface.fill(colors[0], the_square)
                    else:
                        surface.fill(colors[1], the_square)
                    #print "s.xy = %s,%s perm = %s " %(s.pos_x,s.pos_y,s.permanence)
                    text = font.render("%s" % round(s.permanence,3), 1, (255, 50, 200))
                    textpos = (s.pos_x*sq_sz+offset,s.pos_y*sq_sz+offset)
                    surface.blit(text,textpos)
            if event.type == pygame.KEYDOWN:
                
                if event.key == pygame.K_1 or event.key == pygame.K_2 or event.key == pygame.K_3 or event.key == pygame.K_4 or event.key == pygame.K_5 or event.key == pygame.K_6 or event.key == pygame.K_7 or event.key == pygame.K_8 or event.key == pygame.K_9:
                    if event.key == pygame.K_1:
                        if numberLayers>=1:
                            layer = 0       # 0 since the HTM.HTMLayerArray starts at position 0
                            print "layer 1"
                    elif event.key == pygame.K_2:
                        if numberLayers>=2:
                            layer = 1
                            print "layer 2"
                    elif event.key == pygame.K_3:
                        if numberLayers>=3:
                            layer = 2
                            print "layer 3"
                    elif event.key == pygame.K_4:
                        if numberLayers>=4:
                            layer = 3
                            print "layer 4"
                    elif event.key == pygame.K_5:
                        if numberLayers>=5:
                            layer = 4
                            print "layer 5"
                    elif event.key == pygame.K_6:
                        if numberLayers>=6:
                            layer = 5
                            print "layer 6"
                    elif event.key == pygame.K_7:
                        if numberLayers>=7:
                            layer = 6
                            print "layer 7"
                    elif event.key == pygame.K_8:
                        if numberLayers>=8:
                            layer = 7
                            print "layer 8"
                    elif event.key == pygame.K_9:
                        if numberLayers>=9:
                            layer = 8
                            print "layer 9"
                    for row in range(r):           # Draw each row of the board.
                        for col in range(c):       # Run through cols drawing squares
                            the_square = (col*sq_sz, row*sq_sz, sq_sz, sq_sz)
                            if HTM.HTMLayerArray[layer].columns[row][col].activeState==True:
                                surface.fill(colors[0], the_square)
                            else:
                                surface.fill(colors[1], the_square)
                            text = font.render("%s" % HTM.HTMLayerArray[layer].columns[row][col].overlap, 1, (255, 50, 0))
                            textpos = (col*sq_sz+offset,row*sq_sz+offset)
                            surface.blit(text,textpos)
                if event.key == pygame.K_ESCAPE:
                    print "escape"
                    drawing = False
                elif event.key == pygame.K_i:
                    print "i input"
                    for row in range(len(HTM.HTMLayerArray[layer].input)):           # Draw each row of the board.
                        for col in range(len(HTM.HTMLayerArray[layer].input[row])):       # Run through cols drawing squares
                            the_square = (col*sq_sz, row*sq_sz, sq_sz, sq_sz)
                            if HTM.HTMLayerArray[layer].input[row][col]==1:
                                surface.fill(colors[0], the_square)
                            else:
                                surface.fill(colors[1], the_square)
                elif event.key == pygame.K_b:
                    print "b boost"
                    for row in range(r):           # Draw each row of the board.
                        for col in range(c):       # Run through cols drawing squares
                            the_square = (col*sq_sz, row*sq_sz, sq_sz, sq_sz)
                            if HTM.HTMLayerArray[layer].columns[row][col].activeState==True:
                                surface.fill(colors[0], the_square)
                            else:
                                surface.fill(colors[1], the_square)
                            text = font.render("%s" % round(HTM.HTMLayerArray[layer].columns[row][col].boost,3), 1, (255, 50, 200))
                            textpos = (col*sq_sz+offset,row*sq_sz+offset)
                            surface.blit(text,textpos)
                else:
                    for row in range(r):           # Draw each row of the board.
                        for col in range(c):       # Run through cols drawing squares
                            the_square = (col*sq_sz, row*sq_sz, sq_sz, sq_sz)
                            if HTM.HTMLayerArray[layer].columns[row][col].activeState==True:
                                surface.fill(colors[0], the_square)
                            else:
                                surface.fill(colors[1], the_square)
                            text = font.render("%s" % HTM.HTMLayerArray[layer].columns[row][col].overlap, 1, (255, 50, 0))
                            textpos = (col*sq_sz+offset,row*sq_sz+offset)
                            surface.blit(text,textpos)
        pygame.display.flip()
        pygame.event.pump()