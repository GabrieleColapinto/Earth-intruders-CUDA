# Earth intruders - CUDA version

This project is the first of two projects from the assignment "[ADVANCED PROGRAMMING PARADIGMS (781004)](https://www.uah.es/en/estudios/estudios-oficiales/grados/asignatura/Paradigmas-Avanzados-de-Programacion-781004/)" provided by the [University of Alcalá](https://www.uah.es/en/).

To see the second project click [here]().

This game is inspired to the great classic of arcade games "[Space invaders](https://en.wikipedia.org/wiki/Space_Invaders)".

In the original game the player controls a cannon and has to shoot the aliens before they reach the Earth. In this reinterpretation, instead, the player controls a spaceship and has to dodge the incoming aliens. The player is in the bottom row of the matrix and can only move left and right.

When the game starts, the player is asked to enter the size of the matrix and the mode of the game which can be either manual or automatic. In the manual mode, the game waits for the player to input the direction of the movement before proceeding to the next step. In the automatic mode, the game does not wait for the input of the user but it waits 1.5 seconds between each step to allow the user to watch the game. After selecting these parameters, the matrix is initialized and the ship of the player is positioned in the center of the bottom row of the matrix. The ship of the player is identified by the letter 'W'.

In each turn an entire row of aliens is generated in the top row of the stage and the aliens advance downwards by one row. When the aliens reach the bottom of the stage they get destroyed and they increase the score of the player. The generation rates and the rewards of each alien type are in the following table:

## Aliens

| Alien type | Letter | Generation rate | Reward |
| ---------- | :----: | :-------------: | ------ |
| Alien | A | 40% | 5 points |
| Cloud | N | 25% | 25 points |
| Cephalopod | C | 15% | 15 points |
| Destroyer | D | 5% | 5 points |
| Cruiser | R | 13% | 13 points |
| Commander | X | 2% | 100 points + 1 life |

## Shield
During the initialization of the stage, the game generates a random row of shields to allow the player to have a safe space to dodge the incoming aliens. The shield blocks are in the 5<sup>th</sup> row from the bottom of the stage and there cannot be more than 3 consecutive shield blocks. The shield blocks are identified by the letter 'B'.

The shiled blocks can be destroyed by the explosions caused by the collateral effects and by the cruisers and commanders.

## Collateral effects
When the destroyer and the cruiser hit the shield or the Earth they activate a collateral effect.

- The cruiser generates an explosion which either affects the whole row or the whole column in which the explosion is activated.

- The destroyer generates an explosion in a range in the surroundings of the cell in which the explosion is activated.

## Reconversion of the ships
Throughout the game the ships can be reconverted according to the contents of the cells on their left, right, top and bottom. The possible reconversions are the following:

<dl>
<dt><strong>From alien to cloud</strong></dt>
<dd>If an alien is surrounded by other aliens it becomes a cloud and the aliens in its surroundings disappear.</dd>
<dt><strong>From cloud to cephalopod</strong></dt>
<dd>If a cloud is surrounded by aliens it becomes a cephalopod and the aliens in its surroundings disappear.</dd>
<dt><strong>Commander cloud generation</strong></dt>
<dd>Commanders have a probability of 10% to generate a cloud in the surrounding empty cells. There is no need to have all the surrounding cells empty to activate the effect because it fills only the empty cells and does not overwrite the content of the filled cells.</dd>
</dl>

# Versions of the project
- In the basic implementation of the project all the kernels are made to be used in one block. Of course, by doing so we need to consider that the number of threads we can use in the kernels is constrained by the number of threads a block of the video card can handle. For this reason, the number of rows is limited to 50 and the number of columns to 20.

- In the intermediate version of the project the kernels which affect only one row are not changed and the kernels which affect more than one row do it in parallel. We achieve this parallelism referencing the rows of the matrix using the blocks and referencing the cells using the threads.

- In the advanced version of the project kernels use the shared memory.

For more information about the project consult the implementation details contained in the PDF file.