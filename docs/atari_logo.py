           


ATARI_LOGO = """
                                                 .
                                              . ___  .
                                              _/ |.\_  .
                                           . /.  |   \\   .
                                            |    |  . | .
                                           ./    |    \\
                                           |  .  |   . |
                                         . /     |     \\  .
                                          |      |      |   .
                                          / .    |    . \\
                                        .|       |       | .
                                      .  |       |      .|
                                         /.      |       \\  .
                                       ./ - - - -+- - - - \\.
                                       /         |       . \\
                                    .  | .       |         |    .
                                  .   /          I      .  \\
                                     /  .        I        . \\ .
                                   ./ .          I           \\   .
                              .   _/             H           .\_
                                 /               H              \\  .
                          .   __/  .             #             . \__   .
            .        .   ____/.                 J#L               . \____  .
                 .______/    .                  ###                      \______  .
   .    . _______/    .                         ###                    .        \_______  .    .
  _______/    .                                J###L                                .   \_______  
                                                                                            .
     .    ____         ____________________        ____              _____________          ___   
         /    \       |                   |       /    \            |             \        |   |  
        /      \      |_______     _______|      /      \           |    _____     \       |   |  
       /   __   \             |   |             /   __   \          |   |      \    \      |   |  
      /   /  \   \            |   |            /   /  \   \         |   |       |    |     |   |  
      |   |  |   |            |   |            |   |  |   |         |   |______/    /      |   |  
     /   /    \   \           |   |           /   /    \   \        |              /       |   |  
    /   /______\   \          |   |          /   /______\   \       |    _____    /        |   |  
    |   ________   |          |   |          |   ________   |       |   |     \   \        |   |  
   /   /        \   \         |   |         /   /        \   \      |   |      \   \       |   |  
  /   /          \   \        |   |        /   /          \   \     |   |       \   \      |   |  
  |   |          |   |        |   |        |   |          |   |     |   |        |   |     |   |  
  |___|          |___|        |___|        |___|          |___|     |___|        |___|     |___|  

"""

import os
if os.get_terminal_size().columns >= 100:
  print(ATARI_LOGO)
else:
  print('ATARI')