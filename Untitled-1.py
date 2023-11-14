
# Creation of a SparseMatrix class

class SparseMatrix:

    # Creating strucure of matrix

    def __init__(self, row, col):

        self.row = row

        self.col = col

        self.matrix = {}

   

    # Defining a method for setting a value based on given row and column numbers in a sparse matrix

    def set_value(self, row, col, value):

        if row < 0 or row >= self.row or col < 0 or col >= self.col:

            raise ValueError("Invalid row or column index")

        if value is not None:

            self.matrix[(row, col)] = value

        elif (row, col) in self.matrix:

            del self.matrix[(row, col)]

 

    # Defining a method for getting a value based on given row and column numbers in a sparse matrix

    def get_value(self, row, col):

        if row < 0 or row >= self.row or col < 0 or col >= self.col:

            raise ValueError("Invalid row or column index")

        return self.matrix.get((row, col), None)

 

    # Defining a method to multiply the sparse matrix with user provided vectors

    def multiply_with_vector(self, user_vector):

        if len(user_vector) != self.col:

            raise ValueError("Vector length must match the number of columns in the matrix")

 

        result = [0] * self.row

        for row in range(self.row):

            for col in range(self.col):

                if self.get_value(row, col) == None:

                    result[row] += 0 * user_vector[col]

                else:

                    result[row] += self.get_value(row, col) * user_vector[col]

        return result

 

    # Defining a method to add a sparse matrix with other sparse matrix

    def add_matrix(self, other_matrix):

        if self.row != other_matrix.row or self.col != other_matrix.col:

            raise ValueError("Matrix dimensions must match for addition")

 

        result = SparseMatrix(self.row, self.col)

        for row in range(self.row):

            for col in range(self.col):

                movie_title = self.get_value(row, col)

                other_movie_title = other_matrix.get_value(row, col)

                if movie_title != None and other_movie_title != None:

                    value = movie_title + other_movie_title

                    result.set_value(row, col, value)

                elif movie_title:

                    result.set_value(row, col, movie_title)

                elif other_movie_title:

                    result.set_value(row, col, other_movie_title)

        return result

 

    # Defining a method to convert sparse matrix to dense matrix

    def to_dense_matrix(self):

        dense_matrix = [[0] * self.col for _ in range(self.row)]

        for (row, col), movie_title in self.matrix.items():

            dense_matrix[row][col] = movie_title

        return dense_matrix

 

    def __str__(self):

        dense_matrix = self.to_dense_matrix()

        return '\n'.join([' '.join(map(str, row)) for row in dense_matrix])

 

# Creation of a sparse matrix

sparse_matrix = SparseMatrix(3, 3)

 

# Set values at specific positions with values

sparse_matrix.set_value(0, 1, 2.3)

sparse_matrix.set_value(1, 2, 3)

sparse_matrix.set_value(2, 2, 4)


# Get the value at a specific position

print(sparse_matrix.get_value(0, 1))

 

# Create a user vector

user_vector = [1, 1, 1]

 

# Multiply the sparse matrix with the user vector

recommendations = sparse_matrix.multiply_with_vector(user_vector)

print("Multiplication of sparse matrix with user provided vectors:")

print(recommendations)  

 

# creation of another sparse matrix

other_matrix = SparseMatrix(3, 3)

other_matrix.set_value(1, 0, 2)

other_matrix.set_value(2, 2, 4)

 

# Add two sparse matrices

result_matrix = sparse_matrix.add_matrix(other_matrix)

print("Addition of sparse matrix with other sparse matrix:")
print(sparse_matrix) 
print("---------------")
print(other_matrix) 

print("---------------")
print(result_matrix)

 

# Convert the sparse matrix to a dense matrix

dense_matrix = sparse_matrix.to_dense_matrix()

print("Dense matrix of a sparse matrix:")

for row in dense_matrix:

    print(row)

 

 