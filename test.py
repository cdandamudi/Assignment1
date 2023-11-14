import numpy as np

class Test:
    
    def test_verifyCanCreateASparseMatrix(self):
        sparse_matrix = SparseRecommender.SparseMatrix(4, 5)
        assert sparse_matrix != None
        assert (len(sparse_matrix)) == 0

    def test_verifyCanSetValuesToSparseMatrix(self):
        sparse_matrix = SparseRecommender.SparseMatrix(4, 5)
        sparse_matrix.set(0,1,1)
        assert (len(sparse_matrix)) == 1

    def test_verifyCanGetValuesFromSparseMatrix(self):
        sparse_matrix = SparseRecommender.SparseMatrix(4, 5)
        assert sparse_matrix.get(0, 1) == 1
    
    def test_verifyAddMovieFunctionality(self):
        sparse_matrix = SparseRecommender.SparseMatrix(4, 5)
        newMovie_sparseMatrix =  SparseRecommender.SparseMatrix(4, 5)
        sparse_matrix.addNewMovieToSparseMatrix(newMovie_sparseMatrix)
        assert False

    def test_verifyConvertSparseMatrixToDenseMatrix(self):
        sparse_matrix = SparseRecommender.SparseMatrix(4, 5)
        dense_matrix = sparse_matrix.convertSparseMatrixToDenseMatrix()
        assert (isinstance(dense_matrix, np.ndarray))
        assert len(dense_matrix) == sparse_matrix.cols

    def test_verifyRecommend(self):
        sparse_matrix = SparseRecommender.SparseMatrix(4, 5)
        assert len(sparse_matrix.getMovieRecommendations([1,2,3,4,5])) == 5

def getMovieRecommendations(self, userVector):        
        if not len(userVector) == self.columns:
            raise ValueError("Vector length must match the number of columns in the matrix")

        result = np.zeros(len(userVector))  # Initialize the result vector with zeros
        
        for (i, j), value in self.sparseMatrix.items():
            result[i] += value * userVector[j]
        return result