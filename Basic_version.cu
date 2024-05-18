#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// We need these libraries to generate random numbers in a kernel
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>
#include <time.h>
#include <math.h> // We need this library for the ceil function
#include <Windows.h> // We need this library for the sleep function
#include <conio.h> // We need this to read the user input

/*
	The basic implementation of the code is limited by the fact that we have to
	do everything in one block. If we try to use more than 1024 threads the
	curand initializer does not work so we have to limit the number of rows
	and columns according to this constraint.
	Considering that the height of the matrix has to be greater than its width
	we decide to limit the height of the matrix to 50 and the width to 20 for a
	total of at most 1000 threads.
*/
const int MIN_COLUMNAS = 10, MAX_COLUMNAS = 20, MIN_FILAS = 15, MAX_FILAS = 50;
const int SHIELD_ROW = 4, EXPLOSION_RADIUS = 5;
const int WAIT_TIME = 1500; //Milliseconds to wait between each round in automatic mode
int numFilas, numColumnas;
const int SHIELD_PROBABILITY = 15;
char mode;
int lives = 5, score = 0;

// Stage variables
enum modes {
	manual = 'm',
	automatic = 'a'
};
const char DEFAULT_MODE = automatic;

// Cell contents
const int CONTENT_TYPES = 9;
enum cell_contents {
	empty = 0,
	alien = 1,
	cloud = 2,
	cephalopod = 3,
	destroyer = 4,
	cruiser = 5,
	commander = 6,
	shield = 7,
	player = 8
};

const char CELL_LETTERS[CONTENT_TYPES] = {' ', 'A', 'N', 'C', 'D', 'R', 'X', 'B', 'W' };
const int LETTERS_ARRAY_SIZE = sizeof(char) * CONTENT_TYPES;

// Scores
enum scores {
	alien_score = 5,
	cloud_score = 25,
	cephalopod_score = 15,
	destroyer_score = 5,
	cruiser_score = 13,
	commander_score = 100
};

// Boundary values to automatically generate the row of aliens
enum boundary_values {
	empty_val = 0,
	alien_val = 40, // alien = [1, 40]
	cloud_val = 65, // cloud = [41, 65]
	cephalopod_val = 80, // cephalopod = [66, 80]
	destroyer_val = 85, // destroyer = [81, 85]
	cruiser_val = 98, // cruiser = [86, 98]
	commander_val = 100 // commander = [99, 100]
};

enum directions {
	LEFT = 75,
	RIGHT = 77
};

enum collateral_effects {
	ce_destroyer = 1,
	ce_cruiser = 2
};

__constant__ char cell_letters[CONTENT_TYPES];
__constant__ int dev_LEFT = 75, dev_RIGHT = 77;

/*
	The kernel that moves the player requires all the threads to access the same data.
	If we do not use the shared memory every thread in the kernel uses its own value of
	the position of the player.
*/
__shared__ int player_position, direction;

__global__ void manualPlayerMovement(char* dev_Stage_Matrix, int direction, bool* moved, bool* dev_movement_damage, int numColumnas) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	*moved = false;
	*dev_movement_damage = false;

	// We save the position of the player
	if (dev_Stage_Matrix[col] == cell_letters[player]) {
		player_position = col;
	}
	// We wait for all the threads to finish scanning the array
	__syncthreads();

	/*
	  Now we need to first check if the movement is possible and then assess if the
	  player collides with a ship and receives damage.
	  The case in which the player collides with a ship while moving is not described
	  in the assignment so we can decide how to do it.
	  We decide to move the player to the desired location and make it receive damage.

	  The movement is possible if the destination of the player is inside the matrix.
	  The movements which are not allowed are:
		A) player_position == 0 && direction == LEFT
		B) player_position == numColumnas - 1 && direction == RIGHT

	  By the laws of De Morgan: !(A || B) = !A && !B
	*/
	if (!(player_position == 0 && direction == dev_LEFT) &&
		!(player_position == numColumnas - 1 && direction == dev_RIGHT)) {
		/*
			The movement is possible.
			Now we need to set the movement variable to true,
			detect possible collisions and set the damage variable.
		*/
		*moved = true;
		if (direction == dev_LEFT) {
			// It is safe to assume that if a cell is not empty it contains an alien
			if (dev_Stage_Matrix[player_position - 1] != cell_letters[empty]) {
				*dev_movement_damage = true;
			}
			dev_Stage_Matrix[player_position] = cell_letters[empty];
			dev_Stage_Matrix[player_position - 1] = cell_letters[player];
		}
		else {
			// It is safe to assume that if a cell is not empty it contains an alien
			if (dev_Stage_Matrix[player_position + 1] != cell_letters[empty]) {
				*dev_movement_damage = true;
			}
			dev_Stage_Matrix[player_position] = cell_letters[empty];
			dev_Stage_Matrix[player_position + 1] = cell_letters[player];
		}
	}
}

/*
	The kernel to automatically move the player is made in such a way that the player
	always moves.
	If the ship of the player is in a corner it moves inside the matrix or else the
	direction is chosen randomly.
*/
__global__ void automaticPlayerMovement(char* dev_Stage_Matrix, bool* dev_movement_damage, int numColumnas, curandState* state) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int lower = 0, upper = 100, result;

	// We save the position of the player
	if (dev_Stage_Matrix[col] == cell_letters[player]) {
		player_position = col;
	}
	// We wait for all the threads to finish scanning the array
	__syncthreads();

	if (player_position == 0) {
		//The player is in the left corner
		direction = dev_RIGHT;
	}
	else {
		if (player_position == numColumnas - 1) {
			// The player is in the right corner
			direction = dev_LEFT;
		}
		else {
			/*
				We make the thread which indexes the player cell calculate the direction.
				The two directions are either 75 or 77.
				If we calculate a random number and calculate its remainder of the
				division by 2 we get either 0 or 1.
				If we multiply the remainder by 2 we get 0 or 2 so, if we add this value
				to 75 we get either 75 or 77.
			*/
			if (col == player_position) {
				result = curand_uniform(&state[col]) * (upper - lower) + lower;
				direction = dev_LEFT + (result % 2) * 2;
			}
		}
	}

	// We need all the threads to wait for the direction to be calculated
	__syncthreads();


	// Now we need to check if the player receives damage while moving
	if (direction == dev_LEFT) {
		// It is safe to assume that if a cell is not empty it contains an alien
		if (dev_Stage_Matrix[player_position - 1] != cell_letters[empty]) {
			*dev_movement_damage = true;
		}
		dev_Stage_Matrix[player_position] = cell_letters[empty];
		dev_Stage_Matrix[player_position - 1] = cell_letters[player];
	}
	else {
		// It is safe to assume that if a cell is not empty it contains an alien
		if (dev_Stage_Matrix[player_position + 1] != cell_letters[empty]) {
			*dev_movement_damage = true;
		}
		dev_Stage_Matrix[player_position] = cell_letters[empty];
		dev_Stage_Matrix[player_position + 1] = cell_letters[player];
	}
}

__global__ void initCurand(curandState* state, unsigned long seed, unsigned long offset, int numColumnas) {
	int row = threadIdx.x + blockIdx.x * blockDim.x;
	int col = threadIdx.y + blockIdx.y * blockDim.y;
	int id = row * numColumnas + col;
	curand_init(seed, id, offset, &state[id]);
}

/*
	This kernel initializes the matrix filling it with the empty value.
	In the row 0 it inserts the player.
*/
__global__ void matrix_initializer(char* dev_Stage_Matrix, int* dev_Support_Matrix, int initial_player_position, int numFilas, int numColumnas) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int pos;

	if (col < numColumnas) {
		for (int i = numFilas - 1; i >= 0; i--) {
			pos = i * numColumnas + col;
			//Initial position of the player
			if (i == 0 && col == initial_player_position) {
				dev_Stage_Matrix[pos] = cell_letters[player];
			}
			else {
				dev_Stage_Matrix[pos] = cell_letters[empty];
			}
			// The elements of the support matrix need to be set to 0
			dev_Support_Matrix[pos] = 0;
		}
	}
}

__global__ void shieldGenerator(char* dev_Stage_Matrix, curandState* state, int numColumnas) {
	int row = SHIELD_ROW;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int pos = row * numColumnas + col;
	bool A, B, C, D;

	/*
		There cannot be 4 consecutive shield blocks.
		We can have 4 consecutive shield blocks if:
		A) The cell has 3 blocks at its right and they are occupied by a shield
		B) The cell has 1 block at its left, 2 at its right and they are occupied by a shield
		C) The cell has 2 blocks at its left, 1 at its right and they are occupied by a shield
		D) The cell has 3 blocks at its left and they are occupied by a shield

		By the laws of De Morgan:
		!(A || B || C || D) = !A && !B && !C && !D
		Of course, we also need to check if the positions are valid.
	*/
	A = col + 3 < numColumnas &&
		dev_Stage_Matrix[pos + 1] == cell_letters[shield] &&
		dev_Stage_Matrix[pos + 2] == cell_letters[shield] &&
		dev_Stage_Matrix[pos + 3] == cell_letters[shield];

	B = col - 1 >= 0 &&
		col + 2 < numColumnas &&
		dev_Stage_Matrix[pos - 1] == cell_letters[shield] &&
		dev_Stage_Matrix[pos + 1] == cell_letters[shield] &&
		dev_Stage_Matrix[pos + 2] == cell_letters[shield];

	C = col - 2 >= 0 &&
		col + 1 < numColumnas &&
		dev_Stage_Matrix[pos - 2] == cell_letters[shield] &&
		dev_Stage_Matrix[pos - 1] == cell_letters[shield] &&
		dev_Stage_Matrix[pos + 1] == cell_letters[shield];

	D = col - 3 >= 0 &&
		dev_Stage_Matrix[pos - 1] == cell_letters[shield] &&
		dev_Stage_Matrix[pos - 2] == cell_letters[shield] &&
		dev_Stage_Matrix[pos - 3] == cell_letters[shield];

	if (!A && !B && !C && !D) {
		// Now we can generate the shield with a probability of 15%
		int lower = 1, upper = 100, random;
		random = curand_uniform(&state[pos]) * (upper - lower) + lower;
		if (random <= SHIELD_PROBABILITY) {
			dev_Stage_Matrix[pos] = cell_letters[shield];
		}
	}
}

__global__ void addAliens(char* dev_Stage_Matrix, curandState* state, int numFilas, int numColumnas) {
	int row = numFilas - 1;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = row * numColumnas + col;

	/*
		To generate the aliens we generate a random number and we check
		in which range it falls.
		The ranges are defined by the enumeration "boundary_values".
	*/
	int lower = 1, upper = 100, random;
	random = curand_uniform(&state[pos]) * (upper - lower) + lower;

	if (random <= alien_val) {
		dev_Stage_Matrix[pos] = cell_letters[alien];
	}
	else {
		if (random <= cloud_val) {
			dev_Stage_Matrix[pos] = cell_letters[cloud];
		}
		else {
			if (random <= cephalopod_val) {
				dev_Stage_Matrix[pos] = cell_letters[cephalopod];
			}
			else {
				if (random <= destroyer_val) {
					dev_Stage_Matrix[pos] = cell_letters[destroyer];
				}
				else {
					if (random <= cruiser_val) {
						dev_Stage_Matrix[pos] = cell_letters[cruiser];
					}
					else {
						dev_Stage_Matrix[pos] = cell_letters[commander];
					}
				}
			}
		}
	}
}

// Kernel to make the aliens advance
__global__ void advance(char* dev_Stage_Matrix, int* dev_Support_Matrix, int* dev_turn_score, int* dev_turn_lives, bool* dev_advanceDamage, int numFilas, int numColumnas) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < numColumnas) {
		/*
		  We need to memorize the position of possible collateral effects in the support matrix.
		  There are 2 possible collateral effects:
			1 = Destroyer
			2 = Cruiser
		  The support matrix will then be analyzed by the kernels that apply the collateral effects.

		  The variable turn_score contains the total score accumulated in this turn.
		  The variable turn_lives contains the number of lives accumulated in this turn.
		  The variable advanceDamage is set to true if a ship collides with the player.
		*/

		for (int i = 0; i < numFilas - 1; i++) {
			if (i == 0) {
				/*
					We need to delete the ships in row 0 first.
					When a ship is deleted the score is increased.
					Considering that every thread can access score which is located
					in the global memory I need to implement the increment of the score using
					the atomicAdd function to avoid race conditions.
				*/

				// Score increment and collateral effects
				if (dev_Stage_Matrix[col] == cell_letters[alien]) {
					atomicAdd(dev_turn_score, alien_score);
					dev_Stage_Matrix[col] = cell_letters[empty];
				}
				else {
					if (dev_Stage_Matrix[col] == cell_letters[cloud]) {
						atomicAdd(dev_turn_score, cloud_score);
						dev_Stage_Matrix[col] = cell_letters[empty];
					}
					else {
						if (dev_Stage_Matrix[col] == cell_letters[cephalopod]) {
							atomicAdd(dev_turn_score, cephalopod_score);
							dev_Stage_Matrix[col] = cell_letters[empty];
						}
						else {
							if (dev_Stage_Matrix[col] == cell_letters[destroyer]) {
								atomicAdd(dev_turn_score, destroyer_score);
								dev_Support_Matrix[col] = ce_destroyer;
								dev_Stage_Matrix[col] = cell_letters[empty];
							}
							else {
								if (dev_Stage_Matrix[col] == cell_letters[cruiser]) {
									atomicAdd(dev_turn_score, cruiser_score);
									dev_Support_Matrix[col] = ce_cruiser;
									dev_Stage_Matrix[col] = cell_letters[empty];
								}
								else {
									if (dev_Stage_Matrix[col] == cell_letters[commander]) {
										atomicAdd(dev_turn_score, commander_score);
										atomicAdd(dev_turn_lives, 1);
										dev_Stage_Matrix[col] = cell_letters[empty];
									}
								}
							}
						}
					}
				}

				// Advancement of the ships and possible collision with the player
				if (dev_Stage_Matrix[col] == cell_letters[player] && (
					dev_Stage_Matrix[numColumnas + col] == cell_letters[alien] ||
					dev_Stage_Matrix[numColumnas + col] == cell_letters[cloud] ||
					dev_Stage_Matrix[numColumnas + col] == cell_letters[cephalopod] ||
					dev_Stage_Matrix[numColumnas + col] == cell_letters[destroyer] ||
					dev_Stage_Matrix[numColumnas + col] == cell_letters[cruiser] ||
					dev_Stage_Matrix[numColumnas + col] == cell_letters[commander])
					) {
					/*
					* In this case we have a collision. The alien disappears without triggering any collateral
					* effect and the player loses one life.
					*/
					dev_Stage_Matrix[numColumnas + col] = cell_letters[empty];
					*dev_advanceDamage = true;
				}
				else {
					/*
					* In this case we do not have a collision and we can make the
					* spaceships advance.
					*
					* We need to make sure that the player does not get deleted.
					*/
					if (dev_Stage_Matrix[col] != cell_letters[player]) {
						dev_Stage_Matrix[col] = dev_Stage_Matrix[numColumnas + col];
						dev_Stage_Matrix[numColumnas + col] = cell_letters[empty];
					}
				}
			}
			else {
				/*
					If we are in the row below the shield we need to make sure that the
					cell above the one we are considering is not a shield.
					If so we can move its value down and set its value to empty.
				*/
				if (i == SHIELD_ROW - 1) {
					/*
						We need to split the conditions to distinguish the condition of the row from the condition
						of the cell content.
					*/
					if (dev_Stage_Matrix[SHIELD_ROW * numColumnas + col] != cell_letters[shield]) {
						dev_Stage_Matrix[(SHIELD_ROW - 1) * numColumnas + col] = dev_Stage_Matrix[SHIELD_ROW * numColumnas + col];
						dev_Stage_Matrix[SHIELD_ROW * numColumnas + col] = cell_letters[empty];
					}
				}
				else {
					if (i == SHIELD_ROW) {
						/*
							In this case we need to distinguish the cases in which
							the considered cell contains a shield or not.
						*/
						if (dev_Stage_Matrix[SHIELD_ROW * numColumnas + col] == cell_letters[shield]) {
							/*
								In this case we need to check if the cell above the shield contains an alien
								or it is empty.
								In case it is empty nothing happens so we can ignore this case.
								Considering that the cells above the shield can either contain an alien
								or be empty it is safe to assume that if the cell is not empty it contains
								an alien.

								In case the cell above contains an alien we need to check if it can destroy
								the shield and/or activate a collateral effect. Only after checking these two
								things we can delete the alien.
							*/
							if (dev_Stage_Matrix[(SHIELD_ROW + 1) * numColumnas + col] != cell_letters[empty]) {
								/*
									Considering that the ship is destroyed when colliding with the shield,
									the row in which the collateral effect is activated is the one containing
									the shield.
								*/
								if (dev_Stage_Matrix[(SHIELD_ROW + 1) * numColumnas + col] == cell_letters[cruiser]) {
									// The cruiser has a collateral effect and it can destroy the shield.
									dev_Stage_Matrix[SHIELD_ROW * numColumnas + col] = cell_letters[empty];
									dev_Support_Matrix[SHIELD_ROW * numColumnas + col] = ce_cruiser;
								}

								if (dev_Stage_Matrix[(SHIELD_ROW + 1) * numColumnas + col] == cell_letters[destroyer]) {
									// The destroyer does not destroy the shield but activates the collateral effect
									dev_Support_Matrix[SHIELD_ROW * numColumnas + col] = ce_destroyer;
								}

								if (dev_Stage_Matrix[(SHIELD_ROW + 1) * numColumnas + col] == cell_letters[commander]) {
									// The commander can destroy the shield but it does not have a collateral effect
									dev_Stage_Matrix[SHIELD_ROW * numColumnas + col] = cell_letters[empty];
								}
								// Destruction of the alien
								dev_Stage_Matrix[(SHIELD_ROW + 1) * numColumnas + col] = cell_letters[empty];
							}
						}
						else {
							//In this case we can simply make the ships advance
							dev_Stage_Matrix[i * numColumnas + col] = dev_Stage_Matrix[(i + 1) * numColumnas + col];
							dev_Stage_Matrix[(i + 1) * numColumnas + col] = cell_letters[empty];
						}
					}
					else {
						/*
							These rows do not have anything in particular so we can just make the
							contents of the cells advance.
						*/
						dev_Stage_Matrix[i * numColumnas + col] = dev_Stage_Matrix[(i + 1) * numColumnas + col];
						dev_Stage_Matrix[(i + 1) * numColumnas + col] = cell_letters[empty];
					}
				}
			}
		}
	}
}

__global__ void cruiserCollateralEffect(char* dev_Stage_Matrix, int explosionPos, int direction, int numFilas, int numColumnas, bool* dev_cruiser_collateral_damage) {
	int explosionRow = explosionPos / numColumnas;
	int explosionCol = explosionPos % numColumnas;

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int pos;

	if (direction == 0) {
		// We destroy the row
		if (id < numColumnas) {
			pos = explosionRow * numColumnas + id;
			if (dev_Stage_Matrix[pos] == cell_letters[player]) {
				// The player was hit by the explosion
				*dev_cruiser_collateral_damage = true;
			}
			else {
				dev_Stage_Matrix[pos] = cell_letters[empty];
			}
		}
	}
	else {
		// We destroy the column
		if (id < numFilas) {
			pos = numColumnas * id + explosionCol;
			if (dev_Stage_Matrix[pos] == cell_letters[player]) {
				// The player was hit by the explosion
				*dev_cruiser_collateral_damage = true;
			}
			else {
				dev_Stage_Matrix[pos] = cell_letters[empty];
			}
		}
	}
}

__global__ void destroyerCollateralEffect(char* dev_Stage_Matrix, int explosionPos, bool* dev_destroyer_collateral_damage, int numFilas, int numColumnas) {
	int explosionSide = 2 * EXPLOSION_RADIUS + 1;

	// The explosion is applied to the matrix like a mask
	int maskRow = blockIdx.y * blockDim.y + threadIdx.y;
	int maskCol = blockIdx.x * blockDim.x + threadIdx.x;
	int pos;

	// I calculate the coordinates of the center of the explosion
	int explosionCenterRow = explosionPos / numColumnas;
	int explosionCenterCol = explosionPos % numColumnas;

	if (maskRow < explosionSide && maskCol < explosionSide) {
		// I calculate the global coordinates of the thread
		int globalExplosionRow = explosionCenterRow - EXPLOSION_RADIUS + maskRow;
		int globalExplosionCol = explosionCenterCol - EXPLOSION_RADIUS + maskCol;

		if (globalExplosionRow >= 0 && globalExplosionRow < numFilas && globalExplosionCol >= 0 && globalExplosionCol < numColumnas) {
			pos = globalExplosionRow * numColumnas + globalExplosionCol;
			if (dev_Stage_Matrix[pos] == cell_letters[player]) {
				*dev_destroyer_collateral_damage = true;
			}
			else {
				dev_Stage_Matrix[pos] = cell_letters[empty];
			}
		}
	}
}

__global__ void collateralEffectsActivator(char* dev_Stage_Matrix, int* dev_Support_Matrix, int numFilas, int numColumnas, curandState* state, bool* dev_destroyer_collateral_damage, bool* dev_cruiser_collateral_damage) {
	/*
		The collateral effects occur in positions in which the ships can either
		hit the Earth or the shield so it is sufficient to analyze the rows
		0 and SHIELD_ROW.
	*/
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int pos;

	if (col < numColumnas) {
		for (int i = 0; i <= SHIELD_ROW; i += SHIELD_ROW) {
			/*
				First, we need to analyze the support matrix to detect the
				positions in which to apply the collateral effects.
			*/
			pos = i * numColumnas + col;
			if (dev_Support_Matrix[pos] == 1) {
				// Collateral effect of the destroyer
				int explosionSide = 2 * EXPLOSION_RADIUS + 1;
				dim3 THREADS(explosionSide, explosionSide);
				destroyerCollateralEffect << <1, THREADS >> > (dev_Stage_Matrix, pos, dev_destroyer_collateral_damage, numFilas, numColumnas);
				dev_Support_Matrix[pos] = 0;
			}
			else {
				if (dev_Support_Matrix[pos] == 2) {
					/*
						Collateral effect of the cruiser

						To apply the collateral effect of the cruiser we need
						to first generate a direction and then use it to set
						to empty either a row or a column.

						By convention:
							0 = row
							1 = column
					*/

					int lower = 1, upper = 100, random, direction;
					random = curand_uniform(&state[pos]) * (upper - lower) + lower;
					direction = random % 2;

					/*
						We activate the kernel which implements the collateral effect
						of the cruiser.
						We do not know a priori which dimension is bigger but if we
						launch the kernel using the biggest dimension we will be able
						to cover both of them.
					*/
					int longest_dim = (numFilas > numColumnas) ? numFilas : numColumnas;
					cruiserCollateralEffect << <1, longest_dim >> > (dev_Stage_Matrix, pos, direction, numFilas, numColumnas, dev_cruiser_collateral_damage);
					dev_Support_Matrix[pos] = 0;
				}
			}
		}
	}
}

__global__ void reconversion(char* dev_Stage_Matrix, int* dev_Support_Matrix, curandState* state, int numFilas, int numColumnas) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

	if (row < numFilas && col < numColumnas) {
		int pos = row * numColumnas + col;

		/*
			We assign a value to these positions for convenience:
			pos1 = element on the bottom
			pos2 = element on the left
			pos3 = element on the top
			pos4 = element on the right (It is like moving clockwise)
		*/
		int pos1 = (row - 1) * numColumnas + col;
		int pos2 = row * numColumnas + col - 1;
		int pos3 = (row + 1) * numColumnas + col;
		int pos4 = row * numColumnas + col + 1;

		/*
			We need to check if the conversion number 1 is possible.
			We first check if the neighbouring cells are inside the matrix.
			If this condition is verified we can check if those cells contain
			aliens.

			There are 3 types of reconversion:
				1) From alien to cloud
				2) From cloud to cephalopod
				3) The commander generates clouds in its surroundings

			We first analyze the stage matrix to check if the conversions are possible
			and if they are we change the eligibility matrix.
			The cells of the eigibility matrix are filled as follows:
				- We put 1 if we want to convert that cell into a cloud
				- We put 2 if we want to convert that cell into a cephalopod
				- We put 3 if we want to fill that cell with a cloud

			After scanning the matrix we wait for all the threads to finish and
			we can reconvert the matrix.
			While reconverting the stage matrix we also need to change to 0 the values
			of the eligibility matrix.
		*/

		// We scan the matrix
		if (dev_Stage_Matrix[pos] == cell_letters[alien]) {
			if (row - 1 >= 0 && row + 1 < numFilas && col - 1 >= 0 && col + 1 < numColumnas) {
				//The positions are acceptible
				if (dev_Stage_Matrix[pos1] == cell_letters[alien] &&
					dev_Stage_Matrix[pos2] == cell_letters[alien] &&
					dev_Stage_Matrix[pos3] == cell_letters[alien] &&
					dev_Stage_Matrix[pos4] == cell_letters[alien]) {
					/*
						If all the adjacent cells contain an alien we can label this
						position as eligible for reconversion 1.
					*/
					dev_Support_Matrix[pos] = 1;
				}
			}
		}
		else {
			if (dev_Stage_Matrix[pos] == cell_letters[cloud]) {
				if (row - 1 >= 0 && row + 1 < numFilas && col - 1 >= 0 && col + 1 < numColumnas) {
					//The positions are acceptible
					if (dev_Stage_Matrix[pos1] == cell_letters[alien] &&
						dev_Stage_Matrix[pos2] == cell_letters[alien] &&
						dev_Stage_Matrix[pos3] == cell_letters[alien] &&
						dev_Stage_Matrix[pos4] == cell_letters[alien]) {
						/*
							If all the adjacent cells contain an alien we can label this
							position as eligible for reconversion 2.
						*/
						dev_Support_Matrix[pos] = 2;
					}
				}
			}
			else {
				if (dev_Stage_Matrix[pos] == cell_letters[commander]) {
					/*
						The conversion number 3 can occur even if the neighbouring
						cells are not all available.
						We first need to generate the random number to determine
						if the commander is generating clouds in its surroundings.
					*/
					int lower = 1, upper = 100, random;
					random = curand_uniform(&state[pos]) * (upper - lower) + lower;
					if (random <= 10) {
						// In this case the reconversion occurs
						if (row - 1 >= 0 && dev_Stage_Matrix[pos1] == cell_letters[empty]) {
							dev_Support_Matrix[pos1] = 3;
						}
						if (col - 1 >= 0 && dev_Stage_Matrix[pos2] == cell_letters[empty]) {
							dev_Support_Matrix[pos2] = 3;
						}
						if (row + 1 < numFilas && dev_Stage_Matrix[pos3] == cell_letters[empty]) {
							dev_Support_Matrix[pos3] = 3;
						}
						if (col + 1 < numColumnas && dev_Stage_Matrix[pos4] == cell_letters[empty]) {
							dev_Support_Matrix[pos4] = 3;
						}
					}
				}
			}
		}
		/*
			Now that we have filled the eligibility matrix we need to wait for all the threads
			to finish scanning the matrix.
		*/
		__syncthreads();

		//Now we can analyze the eligibility matrix and reconvert the stage matrix.
		if (dev_Support_Matrix[pos] == 1) {
			dev_Stage_Matrix[pos] = cell_letters[cloud];
			dev_Support_Matrix[pos] = 0;
			/*
				We need to check if the neighbouring cells still contain aliens
				beacuse threads are executed in parallel and they can interfere with
				one another.
			*/
			if (dev_Stage_Matrix[pos1] == cell_letters[alien]) {
				dev_Stage_Matrix[pos1] = cell_letters[empty];
			}
			if (dev_Stage_Matrix[pos2] == cell_letters[alien]) {
				dev_Stage_Matrix[pos2] = cell_letters[empty];
			}
			if (dev_Stage_Matrix[pos3] == cell_letters[alien]) {
				dev_Stage_Matrix[pos3] = cell_letters[empty];
			}
			if (dev_Stage_Matrix[pos4] == cell_letters[alien]) {
				dev_Stage_Matrix[pos4] = cell_letters[empty];
			}
		}
		else {
			if (dev_Support_Matrix[pos] == 2) {
				dev_Stage_Matrix[pos] = cell_letters[cephalopod];
				dev_Support_Matrix[pos] = 0;

				/*
					We need to check if the neighbouring cells still contain aliens
					beacuse threads are executed in parallel and they can interfere with
					one another.
				*/
				if (dev_Stage_Matrix[pos1] == cell_letters[alien]) {
					dev_Stage_Matrix[pos1] = cell_letters[empty];
				}
				if (dev_Stage_Matrix[pos2] == cell_letters[alien]) {
					dev_Stage_Matrix[pos2] = cell_letters[empty];
				}
				if (dev_Stage_Matrix[pos3] == cell_letters[alien]) {
					dev_Stage_Matrix[pos3] = cell_letters[empty];
				}
				if (dev_Stage_Matrix[pos4] == cell_letters[alien]) {
					dev_Stage_Matrix[pos4] = cell_letters[empty];
				}
			}
			else {
				if (dev_Support_Matrix[pos] == 3) {
					/*
						In this case we do not need to check the status of the
						neighbouring cells because this reconversion is the only
						one which can put a ship in an empty cell.
					*/
					dev_Stage_Matrix[pos] = cell_letters[cloud];
					dev_Support_Matrix[pos] = 0;
				}
			}
		}
	}
}

void print_matrix(char* matrix) {
	for (int i = numFilas - 1; i >= 0; i--) {
		for (int j = 0; j < numColumnas; j++) {
			printf("%c ", matrix[i * numColumnas + j]);
		}
		printf("\n");
	}
	printf("\nLives = %d\n\nScore = %d", lives, score);
}

int main() {
	curandState* d_state;

	int h_turn_score, h_turn_lives;
	int* dev_turn_score, * dev_turn_lives;

	bool h_advance_damage, h_destroyer_collateral_damage, h_cruiser_collateral_damage, h_movement_damage;
	bool* dev_advance_damage, * dev_destroyer_collateral_damage, * dev_cruiser_collateral_damage, * dev_movement_damage;

	int chr1, h_direction, * dev_direction;
	bool h_moved, * dev_moved;

	// Input of the game settings
	printf("Insert the game settings\n\nInsert the number of rows of the matrix (min = %d, max = %d): ", MIN_FILAS, MAX_FILAS);
	scanf("%d", &numFilas);
	if (numFilas < MIN_FILAS) {
		printf("\nThe minimum number of rows is %d. The number of rows has been set to this value.", MIN_FILAS);
		numFilas = MIN_FILAS;
	}
	else {
		if (numFilas > MAX_FILAS) {
			printf("\nThe maximum number of rows is %d. The number of rows has been set to this value.", MAX_FILAS);
			numFilas = MAX_FILAS;
		}
	}

	printf("\nInsert the number of columns of the matrix (min = %d, max = %d): ", MIN_COLUMNAS, MAX_COLUMNAS);
	scanf("%d", &numColumnas);
	if (numColumnas < MIN_COLUMNAS) {
		printf("\nThe minimum number of columns is %d. The number of columns has been set to this value.", MIN_COLUMNAS);
		numColumnas = MIN_COLUMNAS;
	}
	else {
		if (numColumnas > MAX_COLUMNAS) {
			printf("\nThe maximum number of columns is %d. The number of columns has been set to this value.", MAX_COLUMNAS);
			numColumnas = MAX_COLUMNAS;
		}
	}

	printf("\nSelect the mode of the game. Insert '%c' for manual and '%c' for automatic (default = %c): ", manual, automatic, DEFAULT_MODE);
	scanf(" %c", &mode); //Without the blank space the instruction does not read the user input.
	if (mode != automatic && mode != manual) {
		printf("\nThe selected mode is not valid. The mode has been set to automatic by default.");
		mode = DEFAULT_MODE;
	}

	const int initial_player_position = (int)ceil(numColumnas / 2);

	// Matrices sizes
	const int MATRIX_DIMENSION = numColumnas * numFilas;
	const size_t MAIN_MATRIX_SIZE = sizeof(char*) * MATRIX_DIMENSION;
	const size_t SUPPORT_MATRIX_SIZE = sizeof(int*) * MATRIX_DIMENSION;
	const size_t RAND_STATE_MATRIX_SIZE = sizeof(curandState) * MATRIX_DIMENSION;

	//Main matrix of the game
	char* h_Stage_Matrix = (char*)malloc(MAIN_MATRIX_SIZE);
	char* dev_Stage_Matrix;

	//Support matrix
	int* h_Support_Matrix = (int*)malloc(SUPPORT_MATRIX_SIZE);
	int* dev_Support_Matrix;

	// We allocate the memory in the device
	cudaMalloc(&dev_Stage_Matrix, MAIN_MATRIX_SIZE);
	cudaMalloc(&dev_Support_Matrix, SUPPORT_MATRIX_SIZE);

	cudaMalloc(&d_state, RAND_STATE_MATRIX_SIZE);

	cudaMalloc(&dev_turn_score, sizeof(int));
	cudaMalloc(&dev_turn_lives, sizeof(int));

	cudaMalloc(&dev_movement_damage, sizeof(bool));
	cudaMalloc(&dev_advance_damage, sizeof(bool));
	cudaMalloc(&dev_destroyer_collateral_damage, sizeof(bool));
	cudaMalloc(&dev_cruiser_collateral_damage, sizeof(bool));

	cudaMalloc(&dev_moved, sizeof(bool));
	cudaMalloc(&dev_direction, sizeof(int));

	cudaMemcpyToSymbol(cell_letters, &CELL_LETTERS, LETTERS_ARRAY_SIZE);

	// Initialize cuRAND
	dim3 THREADS(numFilas, numColumnas);
	initCurand << <1, THREADS >> > (d_state, time(NULL), 0, numColumnas);

	// We transfer the data to the device (CPU -> GPU)
	cudaMemcpy(dev_Stage_Matrix, h_Stage_Matrix, MAIN_MATRIX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Support_Matrix, h_Support_Matrix, SUPPORT_MATRIX_SIZE, cudaMemcpyHostToDevice);

	cudaMemcpy(dev_turn_score, &h_turn_score, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_turn_lives, &h_turn_lives, sizeof(int), cudaMemcpyHostToDevice);

	// We generate the initial matrix
	matrix_initializer << <1, numColumnas >> > (dev_Stage_Matrix, dev_Support_Matrix, initial_player_position, numFilas, numColumnas);
	shieldGenerator << <1, numColumnas >> > (dev_Stage_Matrix, d_state, numColumnas);
	addAliens << <1, numColumnas >> > (dev_Stage_Matrix, d_state, numFilas, numColumnas);

	// I retrieve the output value
	cudaMemcpy(h_Stage_Matrix, dev_Stage_Matrix, MAIN_MATRIX_SIZE, cudaMemcpyDeviceToHost);

	system("cls");
	printf("\nMatrix =\n");
	print_matrix(h_Stage_Matrix);

	// We decide to wait 1.5 seconds between the iterations in automatic mode
	if (mode == automatic) {
		Sleep(WAIT_TIME);
	}

	while (lives > 0) {
		h_movement_damage = false;
		h_advance_damage = false;
		h_destroyer_collateral_damage = false;
		h_cruiser_collateral_damage = false;
		h_moved = false;

		h_turn_score = 0;
		h_turn_lives = 0;

		cudaMemcpy(dev_movement_damage, &h_movement_damage, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_advance_damage, &h_advance_damage, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_destroyer_collateral_damage, &h_destroyer_collateral_damage, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_cruiser_collateral_damage, &h_cruiser_collateral_damage, sizeof(bool), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_moved, &h_moved, sizeof(bool), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_turn_score, &h_turn_score, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_turn_lives, &h_turn_lives, sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_Stage_Matrix, h_Stage_Matrix, MAIN_MATRIX_SIZE, cudaMemcpyHostToDevice);

		if (mode == manual) {
			// Manual mode
			printf("\nPress the left or right arrow key...");
			// If the player does not actually move we do not proceed
			while (!h_moved) {
				h_direction = 0;
				/*
					If the user does not press the left or right arrow key
					we do not proceed.
				*/
				while (h_direction != LEFT && h_direction != RIGHT) {
					chr1 = getch();
					if (chr1 == 0xE0) //to check scroll key interrupt
					{
						h_direction = getch();  //to read arrow key
					}
				}


				if (h_direction == LEFT || h_direction == RIGHT) {
					manualPlayerMovement << <1, numColumnas >> > (dev_Stage_Matrix, h_direction, dev_moved, dev_movement_damage, numColumnas);

					// We retrieve the output values
					cudaMemcpy(&h_moved, dev_moved, sizeof(bool), cudaMemcpyDeviceToHost);
				}
			}
		}
		else {
			// Automatic mode
			automaticPlayerMovement<<<1, numColumnas>>>(dev_Stage_Matrix, dev_movement_damage, numColumnas, d_state);
		}

		dim3 RECONVERSION_THREADS(numFilas, numColumnas);
		reconversion<<<1, RECONVERSION_THREADS >>>(dev_Stage_Matrix, dev_Support_Matrix, d_state, numFilas, numColumnas);
		advance << <1, numColumnas >> > (dev_Stage_Matrix, dev_Support_Matrix, dev_turn_score, dev_turn_lives, dev_advance_damage, numFilas, numColumnas);
		collateralEffectsActivator << <1, numColumnas >> > (dev_Stage_Matrix, dev_Support_Matrix, numFilas, numColumnas, d_state, dev_destroyer_collateral_damage, dev_cruiser_collateral_damage);
		addAliens << <1, numColumnas >> > (dev_Stage_Matrix, d_state, numFilas, numColumnas);

		// We retrieve the output values
		cudaMemcpy(h_Stage_Matrix, dev_Stage_Matrix, MAIN_MATRIX_SIZE, cudaMemcpyDeviceToHost);

		cudaMemcpy(&h_movement_damage, dev_movement_damage, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_advance_damage, dev_advance_damage, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_destroyer_collateral_damage, dev_destroyer_collateral_damage, sizeof(bool), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_cruiser_collateral_damage, dev_cruiser_collateral_damage, sizeof(bool), cudaMemcpyDeviceToHost);

		cudaMemcpy(&h_turn_score, dev_turn_score, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&h_turn_lives, dev_turn_lives, sizeof(int), cudaMemcpyDeviceToHost);

		if (h_movement_damage || h_advance_damage || h_destroyer_collateral_damage || h_cruiser_collateral_damage) {
			lives--;
		}

		score += h_turn_score;
		lives += h_turn_lives;

		system("cls");
		printf("\nMatrix =\n");
		print_matrix(h_Stage_Matrix);

		// We decide to wait 1.5 seconds between the iterations in automatic mode
		if (mode == automatic) {
			Sleep(WAIT_TIME);
		}
	}

	// We free the allocated memory
	free(h_Stage_Matrix);
	free(h_Support_Matrix);

	cudaFree(dev_Stage_Matrix);
	cudaFree(dev_Support_Matrix);
	cudaFree(d_state);
	cudaFree(dev_turn_score);
	cudaFree(dev_turn_lives);
	cudaFree(dev_movement_damage);
	cudaFree(dev_advance_damage);
	cudaFree(dev_destroyer_collateral_damage);
	cudaFree(dev_cruiser_collateral_damage);
	cudaFree(dev_moved);
	cudaFree(dev_direction);

	return(0);
}