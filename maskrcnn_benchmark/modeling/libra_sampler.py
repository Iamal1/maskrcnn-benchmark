# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

class LibraSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction,\
         hard_thr=0.0, hard_fraction=1.0, num_intervals=3):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.hard_thr = hard_thr
        self.hard_fraction = hard_fraction
        self.num_intervals = num_intervals

    @staticmethod
    def random_choice(gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]
        

    @staticmethod
    def random_choice_torch(gallery, num):
        """Random select some elements from the gallery.

        Pytorch's implementation
        must assume gallery to be tensor
        with BUG ... not used
        """
        assert len(gallery) >= num
        perm1 = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        
        rand_idx = gallery[perm1]

        return gallery[rand_idx]

    def _sample_pos(self, matched_idxs, num_expected):
        """
        Instance balanced positive sampler
        """
        pos_inds = torch.nonzero(matched_idxs > 0)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            unique_gt_inds = matched_idxs[pos_inds].unique()
            num_gts = len(unique_gt_inds)
            num_per_gt = int(round(num_expected / float(num_gts)) + 1)
            sampled_inds = []
            for i in unique_gt_inds:
                inds = torch.nonzero(matched_idxs == i.item())
                if inds.numel() != 0:
                    inds = inds.squeeze(1)
                else:
                    continue
                if len(inds) > num_per_gt:
                    inds = self.random_choice(inds, num_per_gt)
                sampled_inds.append(inds)
            sampled_inds = torch.cat(sampled_inds)
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(
                    list(set(pos_inds.cpu()) - set(sampled_inds.cpu())))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                ###why is dtype =long---cz nonzero return dtype=torch.int64
                extra_inds = torch.from_numpy(extra_inds).to(
                    matched_idxs.device).long()
                sampled_inds = torch.cat([sampled_inds, extra_inds])
            elif len(sampled_inds) > num_expected:
                sampled_inds = self.random_choice(sampled_inds, num_expected)
            return sampled_inds

    def _sample_via_interval(self, max_overlaps, full_set, num_expected):
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.hard_thr) / self.num_intervals
        per_num_expected = int(num_expected / self.num_intervals)

        sampled_inds = []
        for i in range(self.num_intervals):
            start_iou = self.hard_thr + i * iou_interval
            end_iou = self.hard_thr + (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def _sample_neg(self, matched_idxs, max_overlaps, num_expected):
        """
        max_overlaps(match_quality_matrix): MXN(num_gt,num_proposals)
        """
        neg_inds = torch.nonzero(matched_idxs == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())
            easy_set = set(
                np.where(
                    np.logical_and(max_overlaps >= 0,
                                   max_overlaps < self.hard_thr))[0])
            hard_set = set(np.where(max_overlaps >= self.hard_thr)[0])
            easy_neg_inds = list(easy_set & neg_set)
            hard_neg_inds = list(hard_set & neg_set)

            num_expected_hard = int(num_expected * self.hard_fraction)
            if len(hard_neg_inds) > num_expected_hard:
                if self.num_intervals >= 2:
                    sampled_hard_inds = self._sample_via_interval(
                        max_overlaps, set(hard_neg_inds), num_expected_hard)
                else:
                    sampled_hard_inds = self.random_choice(
                        hard_neg_inds, num_expected_hard)
            else:
                sampled_hard_inds = np.array(hard_neg_inds, dtype=np.int)
            num_expected_easy = num_expected - len(sampled_hard_inds)
            if len(easy_neg_inds) > num_expected_easy:
                sampled_easy_inds = self.random_choice(easy_neg_inds,
                                                       num_expected_easy)
            else:
                sampled_easy_inds = np.array(easy_neg_inds, dtype=np.int)
            sampled_inds = np.concatenate((sampled_easy_inds,
                                           sampled_hard_inds))
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                matched_idxs.device)
            return sampled_inds

    def __call__(self, matched_idxs, match_quality_matrixs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.
            matched quality matrixs : list of tensors containing iou of each gt and proposal pair

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image, max_overlaps_per_image in zip(matched_idxs,match_quality_matrixs):
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # # randomly select positive and negative examples
            # perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            # perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = self._sample_pos(matched_idxs_per_image,num_pos)
            neg_idx_per_image = self._sample_neg(matched_idxs_per_image,max_overlaps_per_image,num_neg)




            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
