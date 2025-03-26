Get project usage:
```sh
kit_project_usage
```

Submit a job:
```sh
sbatch run_app.slurm
```

 Get all submited jobs from one user:
 ```sh
squeue -u uwlat
```

Get output:
```sh
cat logs/output/output_1234567.txt
```

Get Error:
```sh
cat logs/error/error_1234567.txt
```

Get live output:
```sh
tail -f logs/output/output_1234567.txt
```

Get live Error:
```sh
tail -f logs/error/error_1234567.txt
```

Kill a Job:
```sh
scancel 1234567
```

Port forwording:
```sh
ssh -L 5000:hkn0001:5000 uwlat@hk.scc.kit.edu
```

```
ssh -L 5000:<NODE-ID>:5000 uwlat@hk.scc.kit.edu
```
