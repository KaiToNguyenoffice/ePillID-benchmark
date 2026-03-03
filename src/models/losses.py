import torch
import torch.nn as nn
import torch.nn.functional as F
from . import focal_loss
import warnings
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circle Loss  (https://arxiv.org/abs/2002.10857)
# ---------------------------------------------------------------------------
class OnlineCircleLoss(nn.Module):
    """Online Circle Loss that mines positive/negative pairs from a batch."""

    def __init__(self, m=0.25, gamma=256.0, pair_selector=None):
        super().__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()
        self.pair_selector = pair_selector

    def forward(self, embeddings, target, is_front=None, is_ref=None):
        if self.pair_selector is None:
            return None

        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target, is_front, is_ref=is_ref)

        if positive_pairs is None or negative_pairs is None:
            return None

        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        sp = F.cosine_similarity(
            embeddings[positive_pairs[:, 0]],
            embeddings[positive_pairs[:, 1]], eps=1e-12)
        sn = F.cosine_similarity(
            embeddings[negative_pairs[:, 0]],
            embeddings[negative_pairs[:, 1]], eps=1e-12)

        Op = 1 + self.m
        On = -self.m
        delta_p = 1 - self.m
        delta_n = self.m

        ap = torch.clamp_min(Op - sp.detach(), min=0.)
        an = torch.clamp_min(sn.detach() - On, min=0.)

        logit_p = -self.gamma * ap * (sp - delta_p)
        logit_n = self.gamma * an * (sn - delta_n)

        loss = self.soft_plus(
            torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return {
            'loss': loss,
            'circle_pos_sim': sp,
            'circle_neg_sim': sn,
        }


# ---------------------------------------------------------------------------
# Multi-head Loss (updated)
# ---------------------------------------------------------------------------
class MultiheadLoss(nn.Module):
    def __init__(self, n_pilltypes, contrastive_margin, pair_selector,
                 triplet_margin, triplet_selector, use_cosine=False,
                 use_side_labels=True,
                 weights=None, focal_gamma=0.0,
                 circle_params=None):
        super(MultiheadLoss, self).__init__()

        if weights is None:
            weights = {'ce': 1.0, 'arcface': 1.0, 'contrastive': 1.0,
                       'triplet': 1.0, 'focal': 0.0, 'circle': 0.0}

        self.n_pilltypes = n_pilltypes
        self.use_side_labels = use_side_labels
        self.weights = weights
        self.contrastive_loss = OnlineContrastiveLoss(contrastive_margin, pair_selector, use_cosine)
        self.triplet_loss = OnlineTripletLoss(triplet_margin, triplet_selector, use_cosine)

        if weights.get('focal', 0.0) > 0.0:
            logger.info("Using focal loss gamma=%s weight=%s", focal_gamma, weights['focal'])
            self.focal_loss = focal_loss.FocalLossWithOutOneHot(gamma=focal_gamma)

        if weights.get('circle', 0.0) > 0.0:
            cp = circle_params or {}
            logger.info("Using circle loss m=%s gamma=%s weight=%s",
                        cp.get('m', 0.25), cp.get('gamma', 256.0), weights['circle'])
            self.circle_loss = OnlineCircleLoss(
                m=cp.get('m', 0.25),
                gamma=cp.get('gamma', 256.0),
                pair_selector=pair_selector)

    def forward(self, outputs, target, is_front=None, is_ref=None):
        if not self.use_side_labels:
            is_front = None

        emb = outputs['emb']
        if emb.is_cuda:
            device = emb.get_device()
        else:
            device = torch.device('cpu')

        losses = {}
        weighted_loss = torch.zeros(1, dtype=torch.float).to(device)

        # --- Metric losses (computed on original labels) ---
        if self.weights.get('contrastive', 0.0) > 0.0:
            contrastive = self.contrastive_loss(emb, target, is_front=is_front, is_ref=is_ref)
            if contrastive is not None:
                contrastive['contrastive'] = contrastive.pop('loss')
                losses.update(contrastive)
                weighted_loss += contrastive['contrastive'] * self.weights['contrastive']

        if self.weights.get('triplet', 0.0) > 0.0:
            triplet = self.triplet_loss(emb, target, is_front=is_front, is_ref=is_ref)
            if triplet is not None:
                triplet['triplet'] = triplet.pop('loss')
                losses.update(triplet)
                weighted_loss += triplet['triplet'] * self.weights['triplet']

        if self.weights.get('circle', 0.0) > 0.0:
            circle = self.circle_loss(emb, target, is_front=is_front, is_ref=is_ref)
            if circle is not None:
                circle['circle'] = circle.pop('loss')
                losses.update(circle)
                weighted_loss += circle['circle'] * self.weights['circle']

        losses['metric_loss'] = weighted_loss.clone().detach()

        # --- Classification losses (side-shifted labels) ---
        if is_front is not None:
            target = target.clone().detach()
            target[~(is_front.bool())] += self.n_pilltypes

        if self.weights.get('ce', 0.0) > 0.0:
            losses['ce'] = F.cross_entropy(outputs['logits'], target, reduction='mean')
            weighted_loss += losses['ce'] * self.weights['ce']

        if self.weights.get('focal', 0.0) > 0.0:
            losses['focal'] = self.focal_loss(outputs['logits'], target)
            weighted_loss += losses['focal'] * self.weights['focal']

        if self.weights.get('arcface', 0.0) > 0.0:
            losses['arcface'] = F.cross_entropy(outputs['arcface_logits'], target, reduction='mean')
            weighted_loss += losses['arcface'] * self.weights['arcface']

        losses['loss'] = weighted_loss

        return losses


# ---------------------------------------------------------------------------
# Legacy / offline loss classes (kept for compatibility)
# ---------------------------------------------------------------------------
class ContrastiveLoss(nn.Module):
    """Offline contrastive loss on pre-computed pairs."""

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1).clamp(min=1e-12).sqrt()
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - distances.sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """Offline triplet loss on pre-computed triplets."""

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1).clamp(min=1e-12).sqrt()
        distance_negative = (anchor - negative).pow(2).sum(1).clamp(min=1e-12).sqrt()
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


# ---------------------------------------------------------------------------
# Online losses
# ---------------------------------------------------------------------------
class OnlineContrastiveLoss(nn.Module):
    """Online Contrastive loss with hard-negative pair mining."""

    def __init__(self, margin, pair_selector, use_cosine=False):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector
        self.use_cosine = use_cosine

    def forward(self, embeddings, target, is_front=None, is_ref=None):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(
            embeddings, target, is_front, is_ref=is_ref)

        if positive_pairs is None or negative_pairs is None:
            warnings.warn(f"not enough pairs for target {target}")
            return None

        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        losses = []
        if self.use_cosine:
            positive_loss = 1 - F.cosine_similarity(
                embeddings[positive_pairs[:, 0]],
                embeddings[positive_pairs[:, 1]], eps=1e-12)
        else:
            positive_loss = F.pairwise_distance(
                embeddings[positive_pairs[:, 0]],
                embeddings[positive_pairs[:, 1]], eps=1e-12)
        losses.append(positive_loss)

        if self.use_cosine:
            neg_distances = 1 - F.cosine_similarity(
                embeddings[negative_pairs[:, 0]],
                embeddings[negative_pairs[:, 1]], eps=1e-12)
            negative_loss = F.relu(self.margin - neg_distances)
        else:
            negative_loss = F.relu(
                self.margin - F.pairwise_distance(
                    embeddings[negative_pairs[:, 0]],
                    embeddings[negative_pairs[:, 1]], eps=1e-12))
        losses.append(negative_loss)

        loss_outputs = {}
        loss_outputs['loss'] = torch.cat(losses, dim=0).mean()
        all_pairs = torch.cat([positive_pairs, negative_pairs])
        loss_outputs['embeddings'] = (embeddings[all_pairs[:, 0]], embeddings[all_pairs[:, 1]])
        loss_outputs['contrastive_targets'] = torch.cat([
            torch.ones(len(positive_pairs)),
            torch.zeros(len(negative_pairs))
        ], dim=0)
        loss_outputs['contrastive_pos_distances'] = positive_loss
        loss_outputs['contrastive_neg_distances'] = negative_loss
        loss_outputs['contrastive_distances'] = torch.cat([
            positive_loss, negative_loss], dim=0)

        return loss_outputs


class OnlineTripletLoss(nn.Module):
    """Online Triplet loss with hard-negative triplet mining."""

    def __init__(self, margin, triplet_selector, use_cosine=False):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector
        self.use_cosine = use_cosine

    def forward(self, embeddings, target, is_front=None, is_ref=None):
        triplets = self.triplet_selector.get_triplets(
            embeddings, target, is_front=is_front, is_ref=is_ref)

        if triplets is None:
            warnings.warn(f"not enough triplets for target {target}")
            return None

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        if self.use_cosine:
            ap_distances = 1 - F.cosine_similarity(
                embeddings[triplets[:, 0]], embeddings[triplets[:, 1]], eps=1e-12)
            an_distances = 1 - F.cosine_similarity(
                embeddings[triplets[:, 0]], embeddings[triplets[:, 2]], eps=1e-12)
        else:
            ap_distances = F.pairwise_distance(
                embeddings[triplets[:, 0]], embeddings[triplets[:, 1]], eps=1e-12)
            an_distances = F.pairwise_distance(
                embeddings[triplets[:, 0]], embeddings[triplets[:, 2]], eps=1e-12)

        losses = F.relu(ap_distances - an_distances + self.margin)

        loss_outputs = {
            'loss': losses.mean(),
            'triplet_pos_distances': ap_distances,
            'triplet_neg_distances': an_distances,
            'triplet_distances': torch.cat([ap_distances, an_distances], dim=0),
            'triplet_targets': torch.cat([
                torch.ones(len(ap_distances)),
                torch.zeros(len(an_distances))
            ], dim=0),
        }

        return loss_outputs


if __name__ == '__main__':
    from metric_utils import RandomNegativeTripletSelector, HardNegativePairSelector
    import numpy as np

    D_in = 128

    labels = torch.from_numpy(np.array([0] * 7 + [1] * 4))
    is_ref = torch.from_numpy(np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1]))
    embeddings = torch.randn(len(labels), D_in)
    metric_margin = 1

    print('-' * 20 + ' Triplet with is_ref')
    tri_criterion = OnlineTripletLoss(
        metric_margin, RandomNegativeTripletSelector(metric_margin), use_cosine=False)
    euc_loss_out = tri_criterion(embeddings, labels, is_ref=is_ref)
    tri_criterion.use_cosine = True
    cos_loss_out = tri_criterion(embeddings, labels, is_ref=is_ref)
    print(f"Triplet: Euclidean={euc_loss_out['loss']}, Cosine={cos_loss_out['loss']}")

    print('-' * 20 + ' Contrastive with is_ref')
    cnt_criterion = OnlineContrastiveLoss(
        metric_margin, HardNegativePairSelector(), use_cosine=False)
    euc_loss_out = cnt_criterion(embeddings, labels, is_ref=is_ref)
    cnt_criterion.use_cosine = True
    cos_loss_out = cnt_criterion(embeddings, labels, is_ref=is_ref)
    print(f"Contrastive: Euclidean={euc_loss_out['loss']}, Cosine={cos_loss_out['loss']}")

    print('-' * 20 + ' Circle Loss')
    circle_criterion = OnlineCircleLoss(m=0.25, gamma=256.0,
                                        pair_selector=HardNegativePairSelector())
    circle_out = circle_criterion(embeddings, labels, is_ref=is_ref)
    print(f"Circle: loss={circle_out['loss']}")
