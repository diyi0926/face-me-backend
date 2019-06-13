from pathlib import Path
import pandas as pd
from PIL import Image
import get_frame
import numpy as np

class FaceMatcher:
    """
    Class to match ground truth images.
    The image file are stored in gt_images/images/
    The image metadata are stored in gt_images/gt_info.csv
    """
    def __init__(self):
        self.gt_root = Path(__file__).parent / 'gt_images'
        self.gt_csv = self.gt_root / 'gt_info.csv'
        self.gt_images_dir = self.gt_root / 'images'
        self.record = pd.read_csv(self.gt_csv)

        self.gt_images_path = [
            self.gt_images_dir / image_name
            for image_name in self.record['image_name']]
        self.gt_names = self.record['name']
        self.gt_images = [
            np.array(Image.open(image_path).resize((96, 96)))
            for image_path in self.gt_images_path]
        frontFakes_features = [
            get_frame.generate_frontal_face(image)
            for image in self.gt_images]
        self.front_fakes, front_features = zip(*frontFakes_features)
        self.front_features = [feature.flatten() for feature in front_features]

    def match(self, image_ori, image_fake, image_feature):
        """
        Match input image against groundtruth images.

        Parameters
        ----------
        image_ori : 96x96x3 RGB numpy.ndarray
            Original clipped image
        image_fake : 96x96x3 RGB numpy.ndarray
            Frontalized image
        image_feature : numpy.ndarray
            Feature vector generated alongside the frontalized image.
        :return: fake_face: 96x96x3 RGB numpy array

        Returns
        -------
        matched_image : 96x96x3 RGB numpy.ndarray
            The matched image.
        matched_name : string
            Name of the human in matched image.
        matched_front_fake: 96x96x3 RGB numpy.ndarray
            Fake frontalized image from matched_image
        matched_diff: float
            Closeness between input image and matched image
        """
        image_feature = image_feature.squeeze().flatten()

        feature_diff = np.linalg.norm(self.front_features - image_feature, axis=1)
        match_id = np.argmin(feature_diff)

        (matched_image, matched_name, matched_front_fake, matched_diff
         ) = self.gt_images[match_id], self.gt_names[match_id], self.front_fakes[match_id], feature_diff[match_id]

        return matched_image, matched_name, matched_front_fake, matched_diff

face_matcher = FaceMatcher()
