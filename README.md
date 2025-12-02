# DAPoinTr
Abstract:
Point Transformers (PoinTr) have shown great potential in point cloud completion recently. Nevertheless, effective domain adaptation that improves transferability toward target domains remains unexplored. In this paper, we delve into this topic and empirically discover that direct feature alignment on point Transformer’s CNN backbone only brings limited improvements since it cannot guarantee sequence-wise domain-invariant features in the Transformer. To this end, we propose a pioneering Domain Adaptive Point Transformer (DAPoinTr) framework for point cloud completion. DAPoinTr consists of three key components: Domain Query-based Feature Alignment (DQFA), Point Token-wise Feature alignment (PTFA), and Voted Prediction Consistency (VPC). In particular, DQFA is presented to narrow the global domain gaps from the sequence via the presented domain proxy and domain query at the Transformer encoder and decoder, respectively. PTFA is proposed to close the local domain shiftsby aligning the tokens, i.e., point proxy and dynamic query, at the Transformer encoder and decoder, respectively. VPC is designed to consider different Transformer decoders as multiple of experts (MoE) for ensembled prediction voting and pseudo-label generation. Extensive experiments with visualization on several domain adaptation benchmarks demonstrate the effectiveness and superiority of our DAPoinTr compared with state-of-the-art methods. 
# Method

<img width="2633" height="1299" alt="image" src="https://github.com/user-attachments/assets/9c140922-9c5c-4cb1-ab94-dffbf13d7b2c" />

## 📄 Citation (If you find our work useful, please cite)

```bibtex
@inproceedings{li2025dapointr,
  title={Dapointr: Domain adaptive point transformer for point cloud completion},
  author={Li, Yinghui and Zhou, Qianyu and Gong, Jingyu and Zhu, Ye and Dazeley, Richard and Zhao, Xinkui and Lu, Xuequan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={5},
  pages={5066--5074},
  year={2025}
}
```
# Acknowledgement
This code is based on [PoinTr](https://openaccess.thecvf.com/content/ICCV2021/html/Yu_PoinTr_Diverse_Point_Cloud_Completion_With_Geometry-Aware_Transformers_ICCV_2021_paper.html). The models used for partial and complete shape generation are from 3D-FUTURE, ModelNet, CRN, and real-world point clouds. If you find they are useful, please also cite them in your work.
