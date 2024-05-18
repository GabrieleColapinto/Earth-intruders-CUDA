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
	In this version of the project the kernels which affect a grid are now used
	using tiles. The kernels which affect a single row use a temporary row in the
	shared memory.
*/
const int MIN_COLUMNAS = 10, MIN_FILAS = 15;
const int TILE = 10; //Size of the side of the matrix in the shared memory
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

const char CELL_LETTERS[CONTENT_TYPES] = { ' ', 'A', 'N', 'C', 'D', 'R', 'X', 'B', 'W' };
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

	// Temporary row of character in the shared memory
	extern  __shared__ char tempRow[];

	/*
		We transfer the row 0 of the matrix in the temporary
		row and we wait for every thread to finish this task.
	*/
	tempRow[col] = dev_Stage_Matrix[col];
	__syncthreads();

	// We save the position of the player
	if (tempRow[col] == cell_letters[player]) {
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
			if (tempRow[player_position - 1] != cell_letters[empty]) {
				*dev_movement_damage = true;
			}
			tempRow[player_position] = cell_letters[empty];
			tempRow[player_position - 1] = cell_letters[player];
		}
		else {
			// It is safe to assume that if a cell is not empty it contains an alien
			if (tempRow[player_position + 1] != cell_letters[empty]) {
				*dev_movement_damage = true;
			}
			tempRow[player_position] = cell_letters[empty];
			tempRow[player_position + 1] = cell_letters[player];
		}
	}

	/*
		We wait for every thread to finish generating the shield
		and we transfer the data back to the stage matrix.
	*/
	__syncthreads();
	dev_Stage_Matrix[col] = tempRow[col];

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

	// Temporary row of character in the shared memory
	extern  __shared__ char tempRow[];

	/*
		We transfer the row 0 of the matrix in the temporary
		row and we wait for every thread to finish this task.
	*/
	tempRow[col] = dev_Stage_Matrix[col];
	__syncthreads();

	// We save the position of the player
	if (tempRow[col] == cell_letters[player]) {
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
		if (tempRow[player_position - 1] != cell_letters[empty]) {
			*dev_movement_damage = true;
		}
		tempRow[player_position] = cell_letters[empty];
		tempRow[player_position - 1] = cell_letters[player];
	}
	else {
		// It is safe to assume that if a cell is not empty it contains an alien
		if (tempRow[player_position + 1] != cell_letters[empty]) {
			*dev_movement_damage = true;
		}
		tempRow[player_position] = cell_letters[empty];
		tempRow[player_position + 1] = cell_letters[player];
	}

	/*
		We wait for all the threads to finish generating the
		aliens row and we put the data in the stage matrix.
	*/
	__syncthreads();
	dev_Stage_Matrix[col] = tempRow[col];
}

__global__ void initCurand(curandState* state, unsigned long seed, unsigned long offset, int numFilas, int numColumnas) {
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	if (row < numFilas && col < numColumnas) {
		int id = row * numColumnas + col;
		curand_init(seed, id, offset, &state[id]);
	}
}

/*
	This kernel initializes the matrix filling it with the empty value.
	In the row 0 it inserts the player.
*/
__global__ void matrix_initializer(char* dev_Stage_Matrix, int* dev_Support_Matrix, int initial_player_position, int numFilas, int numColumnas) {

	// We store these parameters to use them more quickly in the future
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// We preallocate the shared memory
	__shared__ char submatrix[TILE][TILE];

	// We calculate the global row, column and positions for this thread
	int row = by * TILE + ty;
	int col = bx * TILE + tx;
	int pos = row * numColumnas + col;

	if (pos == initial_player_position) {
		submatrix[ty][tx] = cell_letters[player];
	}
	else {
		submatrix[ty][tx] = cell_letters[empty];
	}

	/*
		We wait for all the threads to finish and we transfer the data
		into the global memory.
	*/
	__syncthreads();

	if (row < numFilas && col < numColumnas) {
		dev_Stage_Matrix[pos] = submatrix[ty][tx];

		// The elements of the support matrix need to be set to 0
		dev_Support_Matrix[pos] = 0;
	}
}

__global__ void shieldGenerator(char* dev_Stage_Matrix, curandState* state, int numColumnas) {
	int row = SHIELD_ROW;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int pos = row * numColumnas + col;
	bool A, B, C, D;

	// Temporary row of character in the shared memory
	extern  __shared__ char tempRow[];

	// We fill the temporary row and wait for every thread to do so
	tempRow[col] = dev_Stage_Matrix[pos];
	__syncthreads();

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
		tempRow[col + 1] == cell_letters[shield] &&
		tempRow[col + 2] == cell_letters[shield] &&
		tempRow[col + 3] == cell_letters[shield];

	B = col - 1 >= 0 &&
		col + 2 < numColumnas &&
		tempRow[col - 1] == cell_letters[shield] &&
		tempRow[col + 1] == cell_letters[shield] &&
		tempRow[col + 2] == cell_letters[shield];

	C = col - 2 >= 0 &&
		col + 1 < numColumnas &&
		tempRow[col - 2] == cell_letters[shield] &&
		tempRow[col - 1] == cell_letters[shield] &&
		tempRow[col + 1] == cell_letters[shield];

	D = col - 3 >= 0 &&
		tempRow[col - 1] == cell_letters[shield] &&
		tempRow[col - 2] == cell_letters[shield] &&
		tempRow[col - 3] == cell_letters[shield];

	if (!A && !B && !C && !D) {
		// Now we can generate the shield with a probability of 15%
		int lower = 1, upper = 100, random;
		random = curand_uniform(&state[pos]) * (upper - lower) + lower;
		if (random <= SHIELD_PROBABILITY) {
			tempRow[col] = cell_letters[shield];
		}
	}

	/*
		We wait for every thread to finish generating the shield
		and we transfer the data back to the stage matrix.
	*/
	__syncthreads();
	dev_Stage_Matrix[pos] = tempRow[col];
}

__global__ void addAliens(char* dev_Stage_Matrix, curandState* state, int numFilas, int numColumnas) {
	int row = numFilas - 1;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = row * numColumnas + col;

	// Temporary row of character in the shared memory
	extern  __shared__ char tempRow[];

	/*
		To generate the aliens we generate a random number and we check
		in which range it falls.
		The ranges are defined by the enumeration "boundary_values".
	*/
	int lower = 1, upper = 100, random;
	random = curand_uniform(&state[pos]) * (upper - lower) + lower;

	if (random <= alien_val) {
		tempRow[col] = cell_letters[alien];
	}
	else {
		if (random <= cloud_val) {
			tempRow[col] = cell_letters[cloud];
		}
		else {
			if (random <= cephalopod_val) {
				tempRow[col] = cell_letters[cephalopod];
			}
			else {
				if (random <= destroyer_val) {
					tempRow[col] = cell_letters[destroyer];
				}
				else {
					if (random <= cruiser_val) {
						tempRow[col] = cell_letters[cruiser];
					}
					else {
						tempRow[col] = cell_letters[commander];
					}
				}
			}
		}
	}

	/*
		We wait for all the threads to finish generating the
		aliens row and we put the data in the stage matrix.
	*/
	__syncthreads();
	dev_Stage_Matrix[pos] = tempRow[col];
}

// Kernel to make the aliens advance
__global__ void advance(char* dev_Stage_Matrix, int* dev_Support_Matrix, int* dev_turn_score, int* dev_turn_lives, bool* dev_advanceDamage, int numFilas, int numColumnas) {
	/*
		The advanced version of this kernel does not use the result matrix.
		Instead, it uses tiles allocated in the shared memory: one for the
		stage matrix and the other for the result matrix.

		We first copy the pieces of the stage matrix in the blocks in the shared
		memory and then we elaborate the result matrix using them.
	*/

	// We store these parameters to use them more quickly in the future
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// We preallocate the shared memory
	__shared__ char stageSubmatrix[TILE][TILE];
	__shared__ char resultSubmatrix[TILE][TILE];

	// We calculate the global row, column and positions for this thread
	int row = by * TILE + ty;
	int col = bx * TILE + tx;
	int pos = row * numColumnas + col;

	//We copy the content of the stage matrix into the shared memory
	stageSubmatrix[ty][tx] = dev_Stage_Matrix[pos];

	/*
	  This function analyzes the stage submatrices and generates the result submatrices.
	  After waiting for all the threads to generate the result submatrices we can
	  copy the content of the result submatrices into the stage matrix in one turn.
	  Using a tiled algorithm introduces the problem of accessing elements external
	  to the tiles. This algorithm moves only vertically so we need to check if the
	  referenced rows are inside the tile and, if not, we need to refer to the
	  global memory.

	  We need to memorize the position of possible collateral effects in the support matrix.
	  There are 2 possible collateral effects:
		1 = Destroyer
		2 = Cruiser
	  The support matrix will then be analyzed by the kernels that apply the collateral effects.

	  The variable turn_score contains the total score accumulated in this turn.
	  The variable turn_lives contains the number of lives accumulated in this turn.
	  The variable advanceDamage is set to true if a ship collides with the player.

	  We mainly refer to the result matrix so that every thread accesses the cell of the
	  result matrix only once. If we refer mainly to the stage matrix, more than one thread
	  would access the cells of the result matrix.
	*/
	if (row == 0) {
		/*
			In this case we do not check that the row above the one considered is included
			in the tile because it is reasonable to assume that the tile has a dimension
			of at least 2x2.
		*/
		if (stageSubmatrix[0][tx] == cell_letters[player]) {
			resultSubmatrix[0][tx] = cell_letters[player];
			//Damage check
			if (stageSubmatrix[1][tx] != cell_letters[empty]) {
				*dev_advanceDamage = true;
			}
		}
		else {
			if (stageSubmatrix[0][tx] == cell_letters[alien]) {
				atomicAdd(dev_turn_score, alien_score);
			}
			else {
				if (stageSubmatrix[0][tx] == cell_letters[cloud]) {
					atomicAdd(dev_turn_score, cloud_score);
				}
				else {
					if (stageSubmatrix[0][tx] == cell_letters[cephalopod]) {
						atomicAdd(dev_turn_score, cephalopod_score);
					}
					else {
						if (stageSubmatrix[0][tx] == cell_letters[destroyer]) {
							atomicAdd(dev_turn_score, destroyer_score);
							dev_Support_Matrix[col] = ce_destroyer;
						}
						else {
							if (stageSubmatrix[0][tx] == cell_letters[cruiser]) {
								atomicAdd(dev_turn_score, cruiser_score);
								dev_Support_Matrix[col] = ce_cruiser;
							}
							else {
								if (stageSubmatrix[0][tx] == cell_letters[commander]) {
									atomicAdd(dev_turn_score, commander_score);
									atomicAdd(dev_turn_lives, 1);
								}
							}
						}
					}
				}
			}
			/*
				These cells do not contain the player and we have calculated the score of the turn.
				We can proceed to copy the values of the cells from the stage matrix.
			*/
			resultSubmatrix[0][tx] = stageSubmatrix[1][tx];
		}
	}
	else {
		if (row == SHIELD_ROW - 1) {
			if (ty + 1 < TILE) { // We access the shared memory
				// We need to make sure that we do not copy the shield
				if (stageSubmatrix[ty + 1][tx] == cell_letters[shield]) {
					resultSubmatrix[ty][tx] = cell_letters[empty];
				}
				else {
					resultSubmatrix[ty][tx] = stageSubmatrix[ty + 1][tx];
				}
			}
			else { // We access the global memory
				// We need to make sure that we do not copy the shield
				if (dev_Stage_Matrix[SHIELD_ROW * numColumnas + col] == cell_letters[shield]) {
					resultSubmatrix[ty][tx] = cell_letters[empty];
				}
				else {
					resultSubmatrix[ty][tx] = dev_Stage_Matrix[SHIELD_ROW * numColumnas + col];
				}
			}
		}
		else {
			if (row == SHIELD_ROW) {
				if (ty + 1 < TILE) {
					/*
						In this case we have to consider that if the cell we are considering contains a shield
						there could be a collision with an alien that triggers something.
					*/
					if (stageSubmatrix[ty][tx] != cell_letters[shield]) {
						// Advance
						resultSubmatrix[ty][tx] = stageSubmatrix[ty + 1][tx];
					}
					else {
						if (stageSubmatrix[ty + 1][tx] == cell_letters[destroyer]) {
							// The destroyer does not destroy the shield but activates the collateral effect
							resultSubmatrix[ty][tx] = cell_letters[shield];
							dev_Support_Matrix[SHIELD_ROW * numColumnas + col] = ce_destroyer;
						}
						else {
							if (stageSubmatrix[ty + 1][tx] == cell_letters[cruiser]) {
								// The cruiser has a collateral effect and it can destroy the shield.
								resultSubmatrix[ty][tx] = cell_letters[empty];
								dev_Support_Matrix[SHIELD_ROW * numColumnas + col] = ce_cruiser;
							}
							else {
								if (stageSubmatrix[ty + 1][tx] == cell_letters[commander]) {
									// The commander can destroy the shield but it does not have a collateral effect
									resultSubmatrix[ty][tx] = cell_letters[empty];
								}
								else {
									// In this case the content of the row above the shield does not destroy it
									resultSubmatrix[ty][tx] = cell_letters[shield];
								}
							}
						}
					}
				}
				else {
					/*
						In this case we have to consider that if the cell we are considering contains a shield
						there could be a collision with an alien that triggers something.
					*/
					if (stageSubmatrix[ty][tx] != cell_letters[shield]) {
						// Advance
						resultSubmatrix[ty][tx] = dev_Stage_Matrix[(row + 1) * numColumnas + col];
					}
					else {
						if (dev_Stage_Matrix[(row + 1) * numColumnas + col] == cell_letters[destroyer]) {
							// The destroyer does not destroy the shield but activates the collateral effect
							resultSubmatrix[ty][tx] = cell_letters[shield];
							dev_Support_Matrix[SHIELD_ROW * numColumnas + col] = ce_destroyer;
						}
						else {
							if (dev_Stage_Matrix[(row + 1) * numColumnas + col] == cell_letters[cruiser]) {
								// The cruiser has a collateral effect and it can destroy the shield.
								resultSubmatrix[ty][tx] = cell_letters[empty];
								dev_Support_Matrix[SHIELD_ROW * numColumnas + col] = ce_cruiser;
							}
							else {
								if (dev_Stage_Matrix[(row + 1) * numColumnas + col] == cell_letters[commander]) {
									// The commander can destroy the shield but it does not have a collateral effect
									resultSubmatrix[ty][tx] = cell_letters[empty];
								}
								else {
									// In this case the content of the row above the shield does not destroy it
									resultSubmatrix[ty][tx] = cell_letters[shield];
								}
							}
						}
					}
				}
			}
			else {
				if (row == numFilas - 1) {
					// We set all the cells to empty
					resultSubmatrix[ty][tx] = cell_letters[empty];
				}
				else {
					if (ty + 1 < TILE) {
						/*
							These rows do not have anything in particular.
							We can simply copy the content from the stage matrix.
						*/
						resultSubmatrix[ty][tx] = stageSubmatrix[ty + 1][tx];
					}
					else {
						resultSubmatrix[ty][tx] = dev_Stage_Matrix[(row + 1) * numColumnas + col];
					}
				}
			}
		}
	}

	/*
		Now we need to wait for all the threads to finish elaborating
		the result submatrices and copy their content into the stage matrix.
		We can do it entirely in parallel.
	*/
	__syncthreads();
	if (col < numColumnas && row < numFilas) {
		dev_Stage_Matrix[row * numColumnas + col] = resultSubmatrix[ty][tx];
	}
}

__global__ void cruiserCollateralEffect(char* dev_Stage_Matrix, int explosionPos, int direction, int numFilas, int numColumnas, bool* dev_cruiser_collateral_damage) {
	int explosionRow = explosionPos / numColumnas;
	int explosionCol = explosionPos % numColumnas;

	int id = blockDim.x * blockIdx.x + threadIdx.x;
	int pos;

	// Temporary row of character in the shared memory
	extern  __shared__ char tempRow[];

	if (direction == 0) {
		// We destroy the row
		if (id < numColumnas) {
			pos = explosionRow * numColumnas + id;

			// We copy the content of the row in the shared memory
			tempRow[id] = dev_Stage_Matrix[pos];
			__syncthreads();

			if (tempRow[id] == cell_letters[player]) {
				// The player was hit by the explosion
				*dev_cruiser_collateral_damage = true;
			}
			else {
				tempRow[id] = cell_letters[empty];
			}

			// We copy the content of the shared memory in the global memory
			__syncthreads();
			dev_Stage_Matrix[pos] = tempRow[id];
		}
	}
	else {
		// We destroy the column
		if (id < numFilas) {
			pos = numColumnas * id + explosionCol;

			// We copy the content of the column in the shared memory
			tempRow[id] = dev_Stage_Matrix[pos];
			__syncthreads();

			if (tempRow[id] == cell_letters[player]) {
				// The player was hit by the explosion
				*dev_cruiser_collateral_damage = true;
			}
			else {
				tempRow[id] = cell_letters[empty];
			}

			// We copy the content of the shared memory in the global memory
			__syncthreads();
			dev_Stage_Matrix[pos] = tempRow[id];
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
		The value row can be either 0 or 1 because this kernel is launched using
		two blocks.
		We can get to all the positions we need if we multiply the value of row
		by SHIELD_ROW.
	*/
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int pos = row * numColumnas * SHIELD_ROW + col;
	int shared_mem_pos = row * numColumnas + col;

	// This variable memorizes the rows 0 and SHIELD_ROW one after the other in a single line
	extern __shared__ int tempRows[];

	// We load the rows of the support matrix in the shared memory
	tempRows[shared_mem_pos] = dev_Support_Matrix[pos];
	__syncthreads();

	if (col < numColumnas && row < numFilas) {
		/*
			First, we need to analyze the support matrix to detect the
			positions in which to apply the collateral effects.
		*/
		if (tempRows[shared_mem_pos] == 1) {
			// Collateral effect of the destroyer
			int explosionSide = 2 * EXPLOSION_RADIUS + 1;
			dim3 BLOCKS(1, explosionSide);
			dim3 THREADS(explosionSide, 1);
			destroyerCollateralEffect << <BLOCKS, THREADS >> > (dev_Stage_Matrix, pos, dev_destroyer_collateral_damage, numFilas, numColumnas);
			tempRows[shared_mem_pos] = 0;
		}
		else {
			if (tempRows[shared_mem_pos] == 2) {
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
				cruiserCollateralEffect << <1, longest_dim, longest_dim * sizeof(char) >> > (dev_Stage_Matrix, pos, direction, numFilas, numColumnas, dev_cruiser_collateral_damage);
				tempRows[shared_mem_pos] = 0;
			}
		}

		// We put the data back in the support matrix
		__syncthreads();
		dev_Support_Matrix[pos] = tempRows[shared_mem_pos];
	}
}

__global__ void reconversion(char* dev_Stage_Matrix, int* dev_Support_Matrix, curandState* state, int numFilas, int numColumnas) {
	// We store these parameters to use them more quickly in the future
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	/*
		We preallocate the shared memory.
		In this kernel we use the shared memory to perform the operations 
		to fill the support matrix.
		We cannot use the shared memory to form the result matrix because
		we would need the threads on the edges of the tile to access
		another tile which is stored in the shared memory of another 
		block and this is not possible.
	*/
	__shared__ char stageSubmatrix[TILE][TILE];

	// We calculate the global row, column and positions for this thread
	int row = by * TILE + ty;
	int col = bx * TILE + tx;
	int pos = row * numColumnas + col;

	// We copy the data in the shared memory
	stageSubmatrix[ty][tx] = dev_Stage_Matrix[pos];
	__syncthreads();

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
		We create variables to store the characters of the neighboring
		cells for convenience.
	*/
	char pos1_char, pos2_char, pos3_char, pos4_char;

	/*
		To check if a conversion is possible we first need to
		check its content and then we need to check the content
		of the neighboring cells. The problem is that the neighboring
		cells can be outside the matrix and, even if they are inside the matrix,
		they cold be outside of the tile. For this reason we need to check the
		correctness of the indices and decide whether to access the shared
		memory or the global memory.

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
	if (stageSubmatrix[ty][tx] == cell_letters[alien]) {
		if (row - 1 >= 0 && row + 1 < numFilas && col - 1 >= 0 && col + 1 < numColumnas) {
			/*
				The positions are acceptible.
				Now we need to check if the neighboring cells
				are inside the tile and assign a value to them accordingly.
			*/
			pos1_char = ty - 1 >= 0 ? stageSubmatrix[ty - 1][tx] : dev_Stage_Matrix[pos1];
			pos2_char = tx - 1 >= 0 ? stageSubmatrix[ty][tx - 1] : dev_Stage_Matrix[pos2];
			pos3_char = ty + 1 < TILE ? stageSubmatrix[ty + 1][tx] : dev_Stage_Matrix[pos3];
			pos4_char = tx + 1 < TILE ? stageSubmatrix[ty][tx + 1] : dev_Stage_Matrix[pos4];

			if (pos1_char == cell_letters[alien] &&
				pos2_char == cell_letters[alien] &&
				pos3_char == cell_letters[alien] &&
				pos4_char == cell_letters[alien]) {
				/*
					If all the adjacent cells contain an alien we can label this
					position as eligible for reconversion 1.
				*/
				dev_Support_Matrix[pos] = 1;
			}
		}
	}
	else {
		if (stageSubmatrix[ty][tx] == cell_letters[cloud]) {
			if (row - 1 >= 0 && row + 1 < numFilas && col - 1 >= 0 && col + 1 < numColumnas) {
				/*
					The positions are acceptible.
					Now we need to check if the neighboring cells
					are inside the tile and assign a value to them accordingly.
				*/
				pos1_char = ty - 1 >= 0 ? stageSubmatrix[ty - 1][tx] : dev_Stage_Matrix[pos1];
				pos2_char = tx - 1 >= 0 ? stageSubmatrix[ty][tx - 1] : dev_Stage_Matrix[pos2];
				pos3_char = ty + 1 < TILE ? stageSubmatrix[ty + 1][tx] : dev_Stage_Matrix[pos3];
				pos4_char = tx + 1 < TILE ? stageSubmatrix[ty][tx + 1] : dev_Stage_Matrix[pos4];

				if (pos1_char == cell_letters[alien] &&
					pos2_char == cell_letters[alien] &&
					pos3_char == cell_letters[alien] &&
					pos4_char == cell_letters[alien]) {
					/*
						If all the adjacent cells contain an alien we can label this
						position as eligible for reconversion 2.
					*/
					dev_Support_Matrix[pos] = 2;
				}
			}
		}
		else {
			if (stageSubmatrix[ty][tx] == cell_letters[commander]) {
				/*
					The conversion number 3 can occur even if the neighboring
					cells are not all available.
					We first need to generate the random number to determine
					if the commander is generating clouds in its surroundings.
				*/
				int lower = 1, upper = 100, random;
				random = curand_uniform(&state[pos]) * (upper - lower) + lower;
				if (random <= 10) {
					// In this case the reconversion occurs
					if (row - 1 >= 0) {
						pos1_char = ty - 1 >= 0 ? stageSubmatrix[ty - 1][tx] : dev_Stage_Matrix[pos1];
						if (pos1_char == cell_letters[empty]) {
							dev_Support_Matrix[pos1] = 3;
						}
					}
					if (col - 1 >= 0) {
						pos2_char = tx - 1 >= 0 ? stageSubmatrix[ty][tx - 1] : dev_Stage_Matrix[pos2];
						if (pos2_char == cell_letters[empty]) {
							dev_Support_Matrix[pos2] = 3;
						}
					}
					if (row + 1 < numFilas) {
						pos3_char = ty + 1 < TILE ? stageSubmatrix[ty + 1][tx] : dev_Stage_Matrix[pos3];
						if (pos3_char == cell_letters[empty]) {
							dev_Support_Matrix[pos3] = 3;
						}
					}
					if (col + 1 < numColumnas) {
						pos4_char = tx + 1 < TILE ? stageSubmatrix[ty][tx + 1] : dev_Stage_Matrix[pos4];
						if (pos4_char == cell_letters[empty]) {
							dev_Support_Matrix[pos4] = 3;
						}
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
	printf("Insert the game settings\n\nInsert the number of rows of the matrix (min = %d): ", MIN_FILAS);
	scanf("%d", &numFilas);
	if (numFilas < MIN_FILAS) {
		printf("\nThe minimum number of rows is %d. The number of rows has been set to this value.", MIN_FILAS);
		numFilas = MIN_FILAS;
	}

	printf("\nInsert the number of columns of the matrix (min = %d): ", MIN_COLUMNAS);
	scanf("%d", &numColumnas);
	if (numColumnas < MIN_COLUMNAS) {
		printf("\nThe minimum number of columns is %d. The number of columns has been set to this value.", MIN_COLUMNAS);
		numColumnas = MIN_COLUMNAS;
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
	const size_t ROW_SIZE = sizeof(char) * numColumnas;

	// Threads and blocks used in the kernels that involve the whole matrix
	const int TILE_BX = (int)ceil((double)numColumnas / TILE);
	const int TILE_BY = (int)ceil((double)numFilas / TILE);
	dim3 TILE_BLOCKS(TILE_BX, TILE_BY);
	dim3 TILE_THREADS(TILE, TILE);

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
	dim3 MATRIX_BLOCKS(1, numFilas);
	dim3 MATRIX_THREADS(numColumnas, 1);
	initCurand << <MATRIX_BLOCKS, MATRIX_THREADS >> > (d_state, time(NULL), 0, numFilas, numColumnas);

	// Blocks to be used in the kernel to activate the collateral effects
	dim3 COLLATERAL_BLOCKS(1, 2);

	// We transfer the data to the device (CPU -> GPU)
	cudaMemcpy(dev_Stage_Matrix, h_Stage_Matrix, MAIN_MATRIX_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Support_Matrix, h_Support_Matrix, SUPPORT_MATRIX_SIZE, cudaMemcpyHostToDevice);

	cudaMemcpy(dev_turn_score, &h_turn_score, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_turn_lives, &h_turn_lives, sizeof(int), cudaMemcpyHostToDevice);

	// We generate the initial matrix
	matrix_initializer << <TILE_BLOCKS, TILE_THREADS >> > (dev_Stage_Matrix, dev_Support_Matrix, initial_player_position, numFilas, numColumnas);
	shieldGenerator << <1, numColumnas, ROW_SIZE >> > (dev_Stage_Matrix, d_state, numColumnas);
	addAliens << <1, numColumnas, ROW_SIZE >> > (dev_Stage_Matrix, d_state, numFilas, numColumnas);

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
					manualPlayerMovement << <1, numColumnas, ROW_SIZE >> > (dev_Stage_Matrix, h_direction, dev_moved, dev_movement_damage, numColumnas);

					// We retrieve the output values
					cudaMemcpy(&h_moved, dev_moved, sizeof(bool), cudaMemcpyDeviceToHost);
				}
			}
		}
		else {
			// Automatic mode
			automaticPlayerMovement << <1, numColumnas, ROW_SIZE >> > (dev_Stage_Matrix, dev_movement_damage, numColumnas, d_state);
		}

		reconversion << <TILE_BLOCKS, TILE_THREADS >> > (dev_Stage_Matrix, dev_Support_Matrix, d_state, numFilas, numColumnas);
		advance << <TILE_BLOCKS, TILE_THREADS >> > (dev_Stage_Matrix, dev_Support_Matrix, dev_turn_score, dev_turn_lives, dev_advance_damage, numFilas, numColumnas);
		collateralEffectsActivator << <COLLATERAL_BLOCKS, MATRIX_THREADS, 2 * ROW_SIZE >> > (dev_Stage_Matrix, dev_Support_Matrix, numFilas, numColumnas, d_state, dev_destroyer_collateral_damage, dev_cruiser_collateral_damage);
		addAliens << <1, numColumnas, ROW_SIZE >> > (dev_Stage_Matrix, d_state, numFilas, numColumnas);

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