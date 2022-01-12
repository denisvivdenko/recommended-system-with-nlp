from typing import List
import pandas as pd

class RecommendationSystem:
    distance_scores = ["cosine", "euclidean"]

    def __init__(self, nearest_neighbors_number: int, 
                items_number: int, 
                distance_score: str = "cosine") -> None:
        self.nearest_neighbors_number = nearest_neighbors_number
        self.items_number = items_number
        if distance_score not in self.distance_scores:
            raise Exception("uknown distance score")
        self.distance_score = distance_score
        self.factorized_matrix: pd.DataFrame = None
        self.users_purchase_data: pd.DataFrame = None
        self.is_ascending_sorting: bool = False

    def train(self, factorized_matrix: pd.DataFrame, users_purchase_data: pd.DataFrame) -> None:
        self.factorized_matrix = factorized_matrix
        self.users_purchase_data = users_purchase_data
        if self.distance_score == "euclidean":
            self.is_ascending_sorting = True

    def predict(self, user_id: int) -> List[str]:
        if not self._is_trained():
            raise Exception("model has not been trained")

        nearest_neighbors: pd.Series = self.factorized_matrix.loc[user_id]
        nearest_neighbors.sort_values(ascending=self.is_ascending_sorting, inplace=True)
        nearest_neighbors = nearest_neighbors.head(self.nearest_neighbors_number).index.values

        nearest_neighbors_purchases = set(self.users_purchase_data[self.users_purchase_data["customer_id"].isin(nearest_neighbors)]["category"])
        customer_purchases = set(self.users_purchase_data[self.users_purchase_data == user_id]["category"])
        return list(nearest_neighbors_purchases.difference(customer_purchases))


    def _is_trained(self) -> bool:
        if (self.factorized_matrix is not None) or (self.users_purchase_data is not None):
            return True

        return False