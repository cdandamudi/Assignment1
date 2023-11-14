import sparse_recommender as sr
import numpy as np
import pytest

class Test_Sparse_Recommender:

    #Helper method to create a sparse matrix in below tests
    def createASparseMatrix(self):
        sparseMatrix = sr.SparseMatrix(4, 5)
        sparseMatrix.setCellValue(0, 1, 1)
        sparseMatrix.setCellValue(1, 0, 2)
        sparseMatrix.setCellValue(1, 2, 3)
        sparseMatrix.setCellValue(2, 1, 4)
        sparseMatrix.setCellValue(2, 3, 5)
        sparseMatrix.setCellValue(3, 0, 6)
        sparseMatrix.setCellValue(3, 2, 7)
        sparseMatrix.setCellValue(3, 4, 8)
        return sparseMatrix

    #Verifies if a sparse matrix can be created or not
    def test_verifyCanCreateASparseMatrix(self):
        sparseMatrix = sr.SparseMatrix(4, 5)
        assert sparseMatrix != None
        assert (len(sparseMatrix)) == 0

    #Verifies  setCellValue method can add value to selected row and column
    def test_verifyCanSetValuesToSparseMatrix(self):
        sparseMatrix = self.createASparseMatrix() #This method has setCellValue method inside it
        assert (len(sparseMatrix)) == 8

    #Verifies getCellValue method returns exact value using row and column
    def test_verifyCanGetValuesFromSparseMatrix(self):
        
        sparseMatrix = self.createASparseMatrix()

        assert sparseMatrix.getCellValue(0, 1) == 1
        assert sparseMatrix.getCellValue(1, 0) == 2
        assert sparseMatrix.getCellValue(1, 2) == 3
        assert sparseMatrix.getCellValue(2, 1) == 4
        assert sparseMatrix.getCellValue(2, 3) == 5
        assert sparseMatrix.getCellValue(3, 0) == 6
        assert sparseMatrix.getCellValue(3, 2) == 7
        assert sparseMatrix.getCellValue(3, 4) == 8

    #Verifies AddNewMovie method to add a new sparse matrix   
    def test_verifyAddMovieFunctionality(self):
        sparseMatrix = self.createASparseMatrix()
        newSparseMatrix = self.createASparseMatrix()
        result = sparseMatrix.addNewMovieToSparseMatrix(newSparseMatrix)
        assert result.getCellValue(0,1) == 2
        assert (result.convertSparseMatrixToDenseMatrix() == [[ 0,  2,  0,  0,  0], [ 4,  0,  6,  0,  0], [ 0,  8,  0, 10,  0], [12,  0, 14,  0, 16]]).all()
         
    #Verifies convertSparseMatrixToDenseMatrix method to convert the sparse matrix to dense matrix
    def test_verifyConvertSparseMatrixToDenseMatrix(self):
        sparseMatrix = self.createASparseMatrix()
        dense_matrix = sparseMatrix.convertSparseMatrixToDenseMatrix()
        assert (isinstance(dense_matrix, np.ndarray))
        assert len(dense_matrix) == sparseMatrix.columns
        assert (dense_matrix == [[0, 1, 0, 0, 0], [2, 0, 3, 0, 0], [0, 4, 0, 5, 0], [6, 0, 7, 0, 8]]).all()

    #Verifies getMovieRecommendations method to multiply a vector with the sparse matrix
    def test_verifyGetMovieRecommendations(self):
        sparseMatrix = self.createASparseMatrix()
        assert (sparseMatrix.getMovieRecommendations([1,2,3,4,5]) == [2,11,28,67,0]).all()
        
    #Verifies if an exception is thrown when trying to set empty values to the matrix
    def test_unableToSetEmptyValuesToMatrix(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.setCellValue(2,3,0)
        assert str(e.value) == "Empty value cannot be set to this object"
        
    #Verifies if an exception is thrown when trying to use negative Row number to set the value
    def test_unableToSetValueToNegativeRowNumber(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.setCellValue(-1,3,2)
        assert str(e.value) == "Row index out of bounds"
        
    #Verifies if an exception is thrown when trying to use out of bounds row number to set the value
    def test_unableToSetValueToRowNumberGreaterThanSize(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.setCellValue(5,3,2)
        assert str(e.value) == "Row index out of bounds"
        
    #Verifies if an exception is thrown when trying to use negative column number to set the value
    def test_unableToSetValueToNegativeColumnNumber(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.setCellValue(1,-3,2)
        assert str(e.value) == "Column index out of bounds"
        
    #Verifies if an exception is thrown when trying to use out of bounds column number to set the value
    def test_unableToSetValueToColumnNumberGreaterThanSize(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.setCellValue(2,9,2)
        assert str(e.value) == "Column index out of bounds"
        
    #Verifies if an exception is thrown when trying to use negative Row number to get the value
    def test_unableToGetValueToNegativeRowNumber(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.getCellValue(-1,3)
        assert str(e.value) == "Row index out of bounds"
        
    #Verifies if an exception is thrown when trying to use out of bounds row number to get the value
    def test_unableToGetValueToRowNumberGreaterThanSize(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.getCellValue(5,3)
        assert str(e.value) == "Row index out of bounds"
        
    #Verifies if an exception is thrown when trying to use negative column number to get the value
    def test_unableToGetValueToNegativeColumnNumber(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.getCellValue(1,-3)
        assert str(e.value) == "Column index out of bounds"
        
    #Verifies if an exception is thrown when trying to use out of bounds column number to get the value
    def test_unableToGetValueToColumnNumberGreaterThanSize(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.getCellValue(2,9)
        assert str(e.value) == "Column index out of bounds"
        
    #Verifies if an exception is thrown when trying to use a vector whose length doesn't match column size of sparse matrix
    def test_verifyExceptionRaised_vectorSizeDoesntMatchColumnSize(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            sparseMatrix.getMovieRecommendations([1,2,3,4,5,6])
        assert str(e.value) == "Vector length must match the number of columns in the matrix"
    
    #Verifies if an exception is thrown when trying to add matricies whose dimensions don't match
    def test_verifyExceptionRaised_dimensionsOfMatricesDontMatch(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = self.createASparseMatrix()
            newSparseMatrix = sr.SparseMatrix(2, 1)
            sparseMatrix.addNewMovieToSparseMatrix(newSparseMatrix)
        assert str(e.value) == "Matrix dimensions must match for addition"
        
    #Verifies if an exception is thrown when trying to add empty matrices
    def test_verifyExceptionRaised_AddNewMovieWithEmptyMatrix(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = sr.SparseMatrix(2, 1)
            newSparseMatrix = sr.SparseMatrix(2, 1)
            sparseMatrix.addNewMovieToSparseMatrix(newSparseMatrix)
        assert str(e.value) == "Matrix is empty to add a new movie"
    
    #Verifies if adding small matrices using addNewMovie method works as expected
    def test_verifyAddNewMovieWorksWithSmallMatrices(self):
        sparseMatrix = sr.SparseMatrix(1, 1)
        sparseMatrix.setCellValue(0,0,1)
        newSparseMatrix = sr.SparseMatrix(1, 1)
        newSparseMatrix.setCellValue(0,0,1)
        assert sparseMatrix.addNewMovieToSparseMatrix(newSparseMatrix).getCellValue(0,0) == 2
   
    #Verifies if an exception is thrown when trying to recommend movies with empty sparse matrix
    def test_verifyExceptionRaised_RecommendMoviesWithEmptySparseMatrix(self):
        with pytest.raises(ValueError) as e:
            sparseMatrix = sr.SparseMatrix(4, 5)
            sparseMatrix.getMovieRecommendations([1,2,3,4,5])
        assert str(e.value) == "Matrix is empty to recommend a movie"

    #Verifies if able to convert small sparse matrix to dense matrix
    def test_verifyConvertToDenseMatrixWorksWithSmallMatrices(self):
        sparseMatrix = sr.SparseMatrix(1, 1)
        sparseMatrix.setCellValue(0,0,1)
        assert (sparseMatrix.convertSparseMatrixToDenseMatrix() == [[1]]).all()
   