import numpy as np

class SparseMatrix:
   
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.sparseMatrix = {}

    #Helper methods to simplify method complexity according to clean code principles
    def isValidRowValue(self, row):
        return row >= 0 and row < self.rows
    
    def isValidColumnValue(self, column):
        return column >= 0 and column < self.columns
    
    def isValidCellValue(self, value):
        return value!=0
    
    def isValidUserVector(self, userVector):
        return len(userVector) == self.columns

    def areDimensionsOfTwoMatricesSame(self, matrix):
        return self.rows == matrix.rows and self.columns == matrix.columns
    
    def isMatrixEmpty(self, matrix):
        return len(matrix) == 0
    
    def add(self, *values):
        return sum(filter(None, values))
    
    def __len__(self):
        return len(self.sparseMatrix)
    
    #Method to set cell value using row and column values
    def setCellValue(self, row, column, value):
        if not self.isValidRowValue(row):
            raise ValueError("Row index out of bounds")
        elif not self.isValidColumnValue(column):
            raise ValueError("Column index out of bounds")
        elif self.isValidCellValue(value):
            self.sparseMatrix[(row, column)] = value
        else: 
            raise ValueError("Empty value cannot be set to this object")

    #Method to get cell value using row and column values
    def getCellValue(self, row, column):
        if not self.isValidRowValue(row):
            raise ValueError("Row index out of bounds")
        elif not self.isValidColumnValue(column):
            raise ValueError("Column index out of bounds")
        return self.sparseMatrix.get((row, column), 0)
    
    #Method to get movie recommendations by multiplying sparse matrix with user vector
    def getMovieRecommendations(self, userVector):        
        if not self.isValidUserVector(userVector):
            raise ValueError("Vector length must match the number of columns in the matrix")
        
        elif self.isMatrixEmpty(self.sparseMatrix):
            raise ValueError("Matrix is empty to recommend a movie")
        
        result = np.zeros(len(userVector))
        
        for (i, j), value in self.sparseMatrix.items():
            result[i] += value * userVector[j]
        return result
   
    #Method to add a new movie by adding sparse matrices
    def addNewMovieToSparseMatrix(self, sparseMatrix):
        if not self.areDimensionsOfTwoMatricesSame(sparseMatrix):
            raise ValueError("Matrix dimensions must match for addition")
        elif self.isMatrixEmpty(self.sparseMatrix) or self.isMatrixEmpty(sparseMatrix):
            raise ValueError("Matrix is empty to add a new movie")
        else:
            result = SparseMatrix(self.rows, self.columns)

            for row in range(self.rows):
                for column in range(self.columns):
                    if not self.getCellValue(row, column) == 0 or not sparseMatrix.getCellValue(row, column) == 0:
                        result.setCellValue(row, column, (self.getCellValue(row, column) + sparseMatrix.getCellValue(row, column)))
        return result

    #Method to convert sparse matrix to dense matrix
    def convertSparseMatrixToDenseMatrix(self):
        denseMatrix = np.zeros((self.rows, self.columns))
        for (self.rows, self.columns), value in self.sparseMatrix.items():
            denseMatrix[self.rows, self.columns] = value
        return denseMatrix