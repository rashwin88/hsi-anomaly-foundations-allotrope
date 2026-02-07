### Some decisions regarding the `FileHelper` Abstraction

One thing that has been bothering me is that we are putting in too much detail in the file helper abstraction, more that what might be necessary. File handling is a grunt operation that needs to have no real knowledge of what product we are using and what the reference template is. These things have to be injected into the file helper. The problem is this:

```python

@property
@abstractmethod
def product(self) -> Product:
    """
    The product from which this file is produced
    """
    pass

@property
@abstractmethod
def template(self) -> Dict[HyperspectralFileComponents, ReferenceDefinition]:
    """
    The reference template corresponding to the product
    """
    pass

```

Of these, the product is of no relevance so I have decided to remove that. The template is essential because it is used to reference specific things in the file structure. But currently I am just looking up a template with in the FileHelper based on the product. Here is an implementation in the case of the `HE5Helper`

```python
# Get the template mappings
self._template: Dict[HyperspectralFileComponents, ReferenceDefinition] = (
    TEMPLATE_MAPPINGS.get(self.product)
)
```

This IMO must be happening outside the Filehelper and all it receives is an injected template which is a dict. I will be making these changes and testing once more. There is a leakage of concepts that relate to geospatial imagery into the FileHelper which can be avoided. The template is still some sort of dictionary, but we can decide what dataset / product maps to which template outside the filehelper.