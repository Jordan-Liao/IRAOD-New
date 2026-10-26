#!/usr/bin/env python3
"""Full-test ROI t-SNE using archived aligned NPZ features, without sampling."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
import tarfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path('/myfile/experiment/iraod/results')
METHODS = ['IRG', 'LPLD', 'SFUT', 'AASFOD', 'SFYOLO', 'F_archived']
PANEL_METHODS = ['IRG', 'LPLD', 'SFUT', 'DRU', 'AASFOD', 'SFYOLO', 'F_archived']
DISPLAY = {'F_archived': 'CGA+VLST (archived F)'}
DOMAINS = {
    'DIOR': ['brightness', 'cloudy', 'contrast'],
    'RSAR': ['chaff', 'gaussian_white_noise', 'point_target', 'noise_suppression',
             'am_noise_horizontal', 'smart_suppression', 'am_noise_vertical'],
}
CLASSES = {
    'RSAR': ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor'],
    'DIOR': ['airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
             'chimney', 'expressway-service-area', 'expressway-toll-station', 'dam',
             'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
             'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'],
}


def expected_ids(dataset):
    if dataset == 'DIOR':
        path = Path('/myfile/dataset/DIOR/ImageSets/Main/test.txt')
        return {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return {p.stem for p in Path('/myfile/dataset/RSAR/test/annfiles').glob('*.txt')}


def discover():
    candidates = {}
    regex = re.compile(r'(?:case|roi)-(DIOR|RSAR)-(.+)-(?:seed_)?42-(.+)-ema\.tar$')
    for directory, _, files in os.walk(ROOT):
        if 'native_test' in directory:
            continue
        for filename in files:
            match = regex.fullmatch(filename)
            if not match:
                continue
            dataset, domain, method = match.groups()
            if method not in METHODS or domain not in DOMAINS[dataset]:
                continue
            path = Path(directory) / filename
            key = (dataset, domain, method)
            if key in candidates:
                raise RuntimeError(f'Ambiguous full-test archive for {key}: {candidates[key]}, {path}')
            candidates[key] = str(path)
    jobs = [dict(dataset=k[0], domain=k[1], method=k[2], archive=v)
            for k, v in sorted(candidates.items())]
    snapshot = ROOT / 'completed_snapshot_20260912T014556Z'
    archived = {}
    with gzip.open(snapshot/'files-67-roi.jsonl.gz', 'rt') as stream:
        for line in stream:
            if '/F/ema/' not in line or 'iraod-aligned-roi-v3-full-test' not in line:
                continue
            record = json.loads(line)
            member = record['archive_member']
            match = re.search(r'iraod-aligned-roi-v3-full-test/(DIOR|RSAR)/([^/]+)/F/ema/',member)
            if not match:
                continue
            dataset, domain = match.groups()
            if domain not in DOMAINS[dataset]:
                continue
            key = (dataset, domain)
            job = archived.setdefault(key, dict(dataset=dataset,domain=domain,method='F_archived',
                archives=set(),member_prefix=member[:match.end()],source_version='archived F; not current LoRA-CGA+VLST'))
            job['archives'].add(str(snapshot/record['asset_name']))
    for job in archived.values():
        job['archives'] = sorted(job['archives'])
        jobs.append(job)
    expected = {(ds,d,m) for ds,domains in DOMAINS.items() for d in domains for m in METHODS}
    found = {(j['dataset'],j['domain'],j['method']) for j in jobs}
    if expected != found:
        raise RuntimeError(f'Missing full-test archive cases: {sorted(expected-found)}')
    return jobs


def load_archive(job):
    start = time.time()
    chunks, labels, scores, image_indices = [], [], [], []
    image_names, empty_images = [], 0
    seen = set()
    digest = hashlib.sha256()
    source_index = None
    for archive_path in job.get('archives',[job.get('archive')]):
      with tarfile.open(archive_path, mode='r:') as archive:
        for member in archive:
            if job.get('member_prefix') and not member.name.startswith(job['member_prefix']):
                continue
            if member.isfile() and member.name.endswith('/index.json'):
                source_index = json.load(archive.extractfile(member))
            if not member.isfile() or not member.name.endswith('.npz'):
                continue
            image_id = Path(member.name).stem
            if image_id in seen:
                raise RuntimeError(f'Duplicate image {image_id} in {archive_path}')
            seen.add(image_id)
            payload = archive.extractfile(member).read()
            digest.update(member.name.encode())
            digest.update(payload)
            with np.load(io.BytesIO(payload), allow_pickle=False) as data:
                x = np.asarray(data['features'], dtype=np.float32)
                y = np.asarray(data['labels'], dtype=np.int32)
                confidence = np.asarray(data['scores'], dtype=np.float32)
                if x.ndim != 2 or len(x) != len(y) or len(y) != len(confidence):
                    raise RuntimeError(f'Unaligned rows: {member.name}')
                if not np.isfinite(x).all() or not np.isfinite(confidence).all():
                    raise RuntimeError(f'Nonfinite input: {member.name}')
                if len(y) and (y.min() < 0 or y.max() >= len(CLASSES[job['dataset']])):
                    raise RuntimeError(f'Invalid labels: {member.name}')
                if 'image_ids' in data and len(data['image_ids']):
                    if not np.all(data['image_ids'].astype(str) == image_id):
                        raise RuntimeError(f'Image ID mismatch: {member.name}')
                if len(x):
                    chunks.append(x)
                    labels.append(y)
                    scores.append(confidence)
                    image_indices.append(np.full(len(x), len(image_names), dtype=np.int32))
                else:
                    empty_images += 1
                image_names.append(image_id)
    expected = expected_ids(job['dataset'])
    missing, extra = sorted(expected-seen), sorted(seen-expected)
    if missing or extra:
        raise RuntimeError(f'Incomplete TEST {job}: missing={len(missing)}, extra={len(extra)}')
    if not chunks:
        raise RuntimeError(f'No ROI features: {job}')
    x = np.concatenate(chunks)
    y = np.concatenate(labels)
    confidence = np.concatenate(scores)
    image_index = np.concatenate(image_indices)
    provenance = {}
    if source_index is not None:
        provenance = {k:v for k,v in source_index.items() if k not in ('records','run')}
        provenance['run'] = {k:v for k,v in source_index.get('run',{}).items() if k != 'image_ids'}
        if source_index.get('n_images',len(seen)) != len(seen):
            raise RuntimeError('Index image count disagrees with loaded features')
        if source_index.get('n_post_nms_detections',len(x)) != len(x):
            raise RuntimeError('Index ROI count disagrees with loaded features')
        if source_index.get('run',{}).get('seed',42) != 42:
            raise RuntimeError('Unexpected source experiment seed')
        if source_index.get('classes',CLASSES[job['dataset']]) != CLASSES[job['dataset']]:
            raise RuntimeError('Source class index order differs from figure legend')
    if job['method']=='F_archived' and not provenance:
        raise RuntimeError('Archived F requires its source index provenance')
    meta = dict(job, expected_images=len(expected), covered_images=len(seen),
                empty_images=empty_images, roi_rows=len(x), feature_dim=x.shape[1],
                input_sha256=digest.hexdigest(), score_min=float(confidence.min()),
                score_max=float(confidence.max()), load_seconds=time.time()-start,
                source_index=provenance)
    print('COVERAGE', job['dataset'], job['domain'], job['method'],
          f'{len(seen)}/{len(expected)} images, {len(x)} ROI rows', flush=True)
    return x, y, confidence, image_index, np.asarray(image_names), meta


def normalize_and_pca(x, y, n_classes):
    # No class labels enter normalization, PCA, neighbors or t-SNE.
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x /= np.maximum(norm, 1e-12)
    total_mean = x.mean(axis=0, dtype=np.float64)
    within = 0.0
    between = 0.0
    counts = []
    for category in range(n_classes):
        subset = x[y == category]
        counts.append(len(subset))
        if len(subset):
            center = subset.mean(axis=0, dtype=np.float64)
            within += np.square(subset-center).sum(dtype=np.float64)
            between += len(subset)*np.square(center-total_mean).sum()
    stats = dict(class_counts=counts, within_scatter_per_roi=within/len(x),
                 between_scatter_per_roi=between/len(x),
                 between_within_ratio=between/max(within, 1e-12))
    # The GPU is used only for a label-independent linear PCA calculation.
    import torch
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            tx = torch.as_tensor(x, device='cuda')
            tx -= tx.mean(dim=0)
            covariance = tx.T @ tx / max(len(x)-1, 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            components = eigenvectors[:, -min(50, x.shape[1]):]
            z = (tx @ components).cpu().numpy()
            stats['pca_variance_ratio'] = float(eigenvalues[-z.shape[1]:].sum()/eigenvalues.sum())
            del tx, covariance, eigenvalues, eigenvectors, components
            torch.cuda.empty_cache()
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(50, x.shape[1]), random_state=42, svd_solver='randomized')
        z = pca.fit_transform(x)
        stats['pca_variance_ratio'] = float(pca.explained_variance_ratio_.sum())
    return np.ascontiguousarray(z, dtype=np.float32), stats


def process_case(job, cache_root, n_jobs):
    from openTSNE import TSNE
    target = Path(cache_root) / job['dataset'] / job['domain'] / job['method']
    target.mkdir(parents=True, exist_ok=True)
    if (target/'embedding.npz').exists() and (target/'metadata.json').exists():
        return str(target)
    started = time.time()
    with threadpool_limits(limits=n_jobs):
        x, y, confidence, image_index, image_names, metadata = load_archive(job)
        xpca, stats = normalize_and_pca(x, y, len(CLASSES[job['dataset']]))
        del x
        print('TSNE_START', job['dataset'], job['domain'], job['method'], xpca.shape, flush=True)
        embedding = TSNE(
            n_components=2, perplexity=50, initialization='pca',
            learning_rate='auto', early_exaggeration=12,
            early_exaggeration_iter=250, n_iter=750,
            negative_gradient_method='fft', neighbors='annoy',
            n_jobs=n_jobs, random_state=42, verbose=False,
        ).fit(xpca)
        metadata.update(stats)
        metadata.update(seed=42, role='ema', perplexity=50, early_iterations=250,
                        iterations=750, normalization='per-row L2', pca_dimensions=xpca.shape[1],
                        compute_threads=n_jobs,
                        points_subsampled=False, label_used_for_embedding=False,
                        kl_divergence=float(embedding.kl_divergence),
                        total_seconds=time.time()-started)
        np.savez_compressed(target/'embedding.npz', embedding=np.asarray(embedding,dtype=np.float32),
                            labels=y, scores=confidence, image_index=image_index,
                            image_names=image_names, pca_features=xpca)
        (target/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('DONE', job['dataset'], job['domain'], job['method'], round(time.time()-started,1), flush=True)
    return str(target)


def colors(dataset):
    if dataset == 'RSAR':
        return ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9']
    return [matplotlib.colors.to_hex(c) for c in plt.get_cmap('tab20').colors]


def coordinate_file(path):
    return path/'embedding.npz' if (path/'embedding.npz').exists() else path/'coordinates.npz'


def render_condition(dataset, domain, cache_root, output_root, compact=False):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':0.5})
    available = []
    for method in METHODS:
        path = Path(cache_root)/dataset/domain/method
        if coordinate_file(path).exists():
            available.append((method,path))
    if not available:
        return
    # Missing planned methods remain visible instead of silently disappearing.
    fig, axes = plt.subplots(2,3 if compact else 4,
        figsize=(7.16,(5.85 if dataset=='DIOR' else 5.3) if compact else (4.95 if dataset=='DIOR' else 4.35)))
    palette = np.asarray([matplotlib.colors.to_rgb(c) for c in colors(dataset)])
    for i, method in enumerate(METHODS if compact else PANEL_METHODS):
        ax=axes.flat[i]
        path = Path(cache_root)/dataset/domain/method
        if not coordinate_file(path).exists():
            ax.set_title(f'({chr(97+i)}) {DISPLAY.get(method,method)}',fontsize=7.5,pad=5)
            ax.text(.5,.5,'ROI features\nnot available',ha='center',va='center',
                    transform=ax.transAxes,fontsize=7,color='#777777')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_color('#BBBBBB')
            continue
        with np.load(coordinate_file(path)) as data:
            z,y=data['embedding'],data['labels']
            order=np.random.default_rng(42).permutation(len(y))
            ax.scatter(z[order,0],z[order,1],c=palette[y[order]],s=0.35,alpha=0.62,
                       linewidths=0,rasterized=True)
        metadata=json.loads((path/'metadata.json').read_text())
        title = 'CGA+VLST\n(archived F)' if method=='F_archived' else method
        ax.set_title(f'({chr(97+i)}) {title}',fontsize=7.5,pad=5)
        ax.set_xlabel(f'{metadata["roi_rows"]:,} ROIs',fontsize=7)
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_aspect('equal',adjustable='datalim')
        for spine in ax.spines.values():spine.set_color('#BBBBBB')
    n_images=json.loads((available[0][1]/'metadata.json').read_text())['covered_images']
    if not compact:
        axes.flat[7].axis('off')
        axes.flat[7].text(.5,.55,f'{n_images:,} / {n_images:,}\nTEST images\n\nNo point subsampling',
                         ha='center',va='center',fontsize=7.5,color='#555555',transform=axes.flat[7].transAxes)
    else:
        fig.text(.5,.012,f'All {n_images:,} test images; no point subsampling. DRU: features unavailable.',
                 ha='center',fontsize=6.4,color='#555555')
    handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=c,markeredgewidth=0,
                    markersize=4,label=name) for c,name in zip(colors(dataset),CLASSES[dataset])]
    fig.legend(handles=handles,loc='lower center',ncol=4 if dataset=='DIOR' else 6,
               frameon=False,fontsize=6.4,columnspacing=.9,handletextpad=.3,
               bbox_to_anchor=(.5,.045 if compact else .018))
    fig.suptitle(f'{dataset} · {domain.replace("_"," ")} · full-test instance features',fontsize=10,y=.99)
    bottom=(.19 if dataset=='DIOR' else .115) if compact else (.18 if dataset=='DIOR' else .10)
    fig.subplots_adjust(left=.035,right=.985,top=.89,bottom=bottom,wspace=.17,hspace=.38)
    target=Path(output_root)/'figures';target.mkdir(parents=True,exist_ok=True)
    if compact:
        target=target/'paper';target.mkdir(exist_ok=True)
    stem=target/f'{dataset}_{domain}_full_test_comparison'
    for ext in ('pdf','svg','png'):
        fig.savefig(str(stem)+'.'+ext,dpi=600 if ext=='png' else 300,bbox_inches='tight')
    plt.close(fig)
    caption=(f'{dataset}, {domain}, seed 42 EMA. Each point is an exported post-NMS ROI feature '
             'colored by its predicted class. All test images and all exported ROI rows are included; '
             'no point or class subsampling is applied. Each panel uses the same L2/PCA/t-SNE '
             'settings, fitted independently. CGA+VLST uses the archived F branch, not the current '
             'LoRA-CGA+VLST evaluated in the result spreadsheet. DRU features are unavailable. '
             'Predicted-label separation is a qualitative property of these features and is not '
             'proof of better detection or semantic accuracy. Axes are arbitrary t-SNE units.')
    Path(str(stem)+'.caption.txt').write_text(caption,encoding='utf-8')


def render_pair(dataset, domain, cache_root, output_root):
    render_condition(dataset,domain,cache_root,output_root)
    render_condition(dataset,domain,cache_root,output_root,compact=True)
    return f'{dataset}/{domain}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--dataset',choices=['DIOR','RSAR','all'],default='all')
    parser.add_argument('--domain',default='all')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    candidates = ([json.loads(p.read_text()) for p in Path(args.cache).glob('*/*/*/metadata.json')]
                  if args.render_only else discover())
    jobs=[j for j in candidates if (args.dataset=='all' or j['dataset']==args.dataset)
          and (args.domain=='all' or j['domain']==args.domain)]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'input_archives.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    if not args.render_only:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            tasks={pool.submit(process_case,j,args.cache,args.threads):j for j in jobs}
            for future in concurrent.futures.as_completed(tasks):
                print('COMPLETED_CASE',future.result(),flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers,4)) as pool:
        figures=[pool.submit(render_pair,dataset,domain,args.cache,output)
                 for dataset,domain in sorted({(j['dataset'],j['domain']) for j in jobs})]
        for future in concurrent.futures.as_completed(figures):
            print('FIGURE_DONE',future.result(),flush=True)
    summaries=[]
    selected_keys={(j['dataset'],j['domain'],j['method']) for j in jobs}
    for path in Path(args.cache).glob('*/*/*/metadata.json'):
        row=json.loads(path.read_text())
        if (row['dataset'],row['domain'],row['method']) not in selected_keys:
            continue
        destination=output/'embedding_data'/row['dataset']/row['domain']/row['method']
        destination.mkdir(parents=True,exist_ok=True)
        (destination/'metadata.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
        with np.load(coordinate_file(path.parent)) as data:
            np.savez_compressed(destination/'coordinates.npz',**{k:data[k] for k in
                ['embedding','labels','scores','image_index','image_names']})
        summaries.append({k:row[k] for k in ['dataset','domain','method','covered_images',
            'expected_images','empty_images','roi_rows','between_within_ratio','input_sha256']})
    if summaries:
        with (output/'coverage_and_feature_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)


if __name__=='__main__':main()
