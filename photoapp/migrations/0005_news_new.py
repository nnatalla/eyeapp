# Generated by Django 4.2.7 on 2023-11-29 12:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("photoapp", "0004_news"),
    ]

    operations = [
        migrations.AddField(
            model_name="news",
            name="new",
            field=models.ImageField(blank=True, null=True, upload_to="news/"),
        ),
    ]