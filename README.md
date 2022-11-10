# Metr-la

---

**Rebuild from**

https://github.com/hazdzz/STGCN

## Info

- Time span: 3/1/2012 - 6/30/2012, 00:00~23:55, 288 intervals
- Number of stations: 207
- Interval: 5min
- Feature: speed

## Data Description

- `vel.pth`: torch tensor
  ```
  shape (n_time, n_vertex, channel), (34272, 207, 1)
  
- `time_index.h5`: pandas table, timestamp for all data, (n_time, 4)
  ```
               time_index timeofday  dayofweek week_name
    0 2012-03-01 00:00:00         0          3  Thursday
    1 2012-03-01 00:05:00         1          3  Thursday
    2 2012-03-01 00:10:00         2          3  Thursday
    3 2012-03-01 00:15:00         3          3  Thursday
    4 2012-03-01 00:20:00         4          3  Thursday
  ```
  
- `adj.pth`: torch tensor, weighted connectivity graph with self loop  
  ```
  shape (n_vertex, n_vertex), (207, 207)
  build by using thresholded Gaussian kernel (Shuman et al., 2013)
  $$
    {{\rm{W}}_{ij}} = \exp ( - \frac{{dist{{({v_i},{v_j})}^2}}}{{{\sigma ^2}}})
  $$
  and if dist > threshold, the value is set to 0,
  Threshold is assigned to 0.1 in this case.
  ```