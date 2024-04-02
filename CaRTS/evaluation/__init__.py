from .metrics import dice_scores, normalized_surface_distances
from .ranking import ranking_scores

metrics_dict = {"dice_scores": dice_scores,
                "normalized_surface_distances": normalized_surface_distances}