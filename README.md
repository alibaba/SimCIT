<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->


<h3 align="center">Gwhere</h3>

  <p align="center">
     Guess Where You Go: Generative Next Point-of-Interest Recommendation in Amap
    <a href="https://github.com/alibaba/SimCIT">View Demo</a>
    &middot;
    <a href="https://github.com/alibaba/SimCIT/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/alibaba/SimCIT/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details> -->



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/alibaba/SimCIT)

Gwhere is an end-to-end generative recommendation framework for next point-of-interest (POI) prediction, designed to overcome the scalability limits of token-based retrieval and the lack of spatial-world understanding in LLMs. It introduces a contrastive item tokenization method that fuses multi-modal signals (text, image, spatial, and collaborative data) into compact, discriminative semantic identifiers (SIDs), enabling efficient large-scale generative retrieval. Built on these SIDs, a spatio-temporal LLM—pretrained on real-world mobility corpora and fine-tuned with a reinforcement learning algorithm (EAKTO)—aligns closely with user behavior.


<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
1. Clone the repository:
   ```bash
   git clone repo_name
   cd repo
   ```
2. Install dependencies:
  * Python 3.9+
  * PyTorch 2.2.0
  * requirements.txt
    ```sh
    pip install -r requirements.txt
    ```

### SID Generation
1. Training the Model

To start distributed training, use the following command:
```
./run_train.sh
```

2. Parameters
- `--state_dict_save_path`: Directory for model outputs.

3. Testing the Model

Use the following command to start testing:

```bash
./run_infer.sh
```

### LLM training

coming soon


<!-- USAGE EXAMPLES -->
<!-- ## Usage
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources. -->
<!-- _For more examples, please refer to the [Documentation]()_ -->
<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->






<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE` for more information.

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT -->
## Contact
If you have any questions or encounter difficulties, we welcome you to contact ours via [GitHub Issues](https://github.com/alibaba/SimCIT/issues). We are dedicated to supporting you in resolving issues related to sid generation, facilitating a robust and efficient setup for your system.
<!-- Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com -->
<!-- Project Link: [https://github.com/selous123/al_sid](https://github.com/selous123/al_sid) -->
<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->


## Citing this work
Please cite the following paper if you find our code helpful.

```
@misc{zhai2025cognitivealignedspatiotemporallargelanguage,
      title={Cognitive-Aligned Spatio-Temporal Large Language Models For Next Point-of-Interest Prediction}, 
      author={Penglong Zhai and Jie Li and Fanyi Di and Yue Liu and Yifang Yuan and Jie Huang and Peng Wu and Sicong Wang and Mingyang Yin and Tingting Hu and Yao Xu and Xin Li},
      year={2025},
      eprint={2510.14702},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.14702}, 
}
```


<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []() -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: asset/framework_gwhere.jpg
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 


