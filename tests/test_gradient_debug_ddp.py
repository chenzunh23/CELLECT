"""Two CPU ranks: rank-zero probes must not enter DDP gradient collectives."""
import tempfile
import json
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tests.test_gradient_debug import TinyModel, make_data, collate, loss_fn
from utils.gradient_debug import GradientDebugger


def worker(rank, path, unused):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=2,
        init_method='file://'+str(Path(path)/'init'),timeout=timedelta(seconds=30))
    try:
        model=DistributedDataParallel(TinyModel(),find_unused_parameters=unused)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        records,data=make_data()
        config=None
        if not unused:
            config=Path(path)/'probes.json'
            if rank==0:
                config.write_text(json.dumps(dict(random_samples_per_dataset=2,datasets={
                    'abell':dict(random_samples=0,probes=[dict(id='halo',name=records[0].name,cohort='abell_halo')])})))
            dist.barrier()
        debugger=GradientDebugger(data,records,collate,Path(path)/'metrics',interval=1,probe_config=config)
        for step in (1,2):
            optimizer.zero_grad()
            a,b=model(torch.tensor([[rank+1.]],dtype=torch.float32))
            (a.square().mean()+b.square().mean()).backward()
            optimizer.step()
            old=[p.grad.clone() for p in model.parameters()]
            debugger.maybe_probe(model.module,optimizer,optimizer_step=step,iteration=step,
                epoch=0,device=torch.device('cpu'),loss_fn=loss_fn)
            for p,grad in zip(model.parameters(),old):torch.testing.assert_close(p.grad,grad)
            vector=torch.cat([p.detach().flatten() for p in model.parameters()])
            gathered=[torch.empty_like(vector) for _ in range(2)]
            dist.all_gather(gathered,vector)
            torch.testing.assert_close(gathered[0],gathered[1],rtol=0,atol=0)
    finally:dist.destroy_process_group()


class DDPGradientTests(unittest.TestCase):
    def test_rank_zero_probe_and_next_training_step(self):
        for unused in (True,False):
            with self.subTest(find_unused_parameters=unused),tempfile.TemporaryDirectory() as tmp:
                mp.spawn(worker,args=(tmp,unused),nprocs=2,join=True)
                rows=(Path(tmp)/'metrics/metrics.jsonl').read_text().splitlines()
                self.assertEqual(len(rows),2)
                self.assertFalse((Path(tmp)/'metrics/errors.jsonl').exists())


if __name__=='__main__':unittest.main()
